import asyncio
import collections
import logging
import copy
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.coordinator.protocol import ConsumerProtocol
from aiokafka.protocol.api import Response
from aiokafka.protocol.commit import (
from aiokafka.protocol.group import (
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.util import create_future, create_task
class GroupCoordinator(BaseCoordinator):
    """
    GroupCoordinator implements group management for single group member
    by interacting with a designated Kafka broker (the coordinator). Group
    semantics are provided by extending this class.

    From a high level, Kafka's group management protocol consists of the
    following sequence of actions:

    1. Group Registration: Group members register with the coordinator
       providing their own metadata
       (such as the set of topics they are interested in).

    2. Group/Leader Selection: The coordinator (one of Kafka nodes) select
       the members of the group and chooses one member (one of client's)
       as the leader.

    3. State Assignment: The leader receives metadata for all members and
       assigns partitions to them.

    4. Group Stabilization: Each member receives the state assigned by the
       leader and begins processing.
       Between each phase coordinator awaits all clients to respond. If some
       do not respond in time - it will revoke their membership

    NOTE: Try to maintain same log messages and behaviour as Java and
          kafka-python clients:

        https://github.com/apache/kafka/blob/0.10.1.1/clients/src/main/java/          org/apache/kafka/clients/consumer/internals/AbstractCoordinator.java
        https://github.com/apache/kafka/blob/0.10.1.1/clients/src/main/java/          org/apache/kafka/clients/consumer/internals/ConsumerCoordinator.java
    """

    def __init__(self, client, subscription, *, group_id='aiokafka-default-group', group_instance_id=None, session_timeout_ms=10000, heartbeat_interval_ms=3000, retry_backoff_ms=100, enable_auto_commit=True, auto_commit_interval_ms=5000, assignors=(RoundRobinPartitionAssignor,), exclude_internal_topics=True, max_poll_interval_ms=300000, rebalance_timeout_ms=30000):
        """Initialize the coordination manager.

        Parameters (see AIOKafkaConsumer)
        """
        self._group_subscription = None
        super().__init__(client, subscription, exclude_internal_topics=exclude_internal_topics)
        self._session_timeout_ms = session_timeout_ms
        self._heartbeat_interval_ms = heartbeat_interval_ms
        self._max_poll_interval = max_poll_interval_ms / 1000
        self._rebalance_timeout_ms = rebalance_timeout_ms
        self._retry_backoff_ms = retry_backoff_ms
        self._assignors = assignors
        self._enable_auto_commit = enable_auto_commit
        self._auto_commit_interval_ms = auto_commit_interval_ms
        self.generation = OffsetCommitRequest.DEFAULT_GENERATION_ID
        self.member_id = JoinGroupRequest[0].UNKNOWN_MEMBER_ID
        self.group_id = group_id
        self._group_instance_id = group_instance_id
        self.coordinator_id = None
        self._performed_join_prepare = False
        self._rejoin_needed_fut = create_future()
        self._coordinator_dead_fut = create_future()
        self._coordination_task = create_task(self._coordination_routine())
        self._heartbeat_task = None
        self._commit_refresh_task = None
        self._pending_exception = None
        self._error_consumed_fut = None
        self._coordinator_lookup_lock = asyncio.Lock()
        self._commit_lock = asyncio.Lock()
        self._next_autocommit_deadline = time.monotonic() + auto_commit_interval_ms / 1000
        self._closing = create_future()

    def _on_metadata_change(self):
        self.request_rejoin()

    async def _send_req(self, request):
        """ Send request to coordinator node. In case the coordinator is not
        ready a respective error will be raised.
        """
        node_id = self.coordinator_id
        if node_id is None:
            raise Errors.GroupCoordinatorNotAvailableError()
        try:
            resp = await self._client.send(node_id, request, group=ConnectionGroup.COORDINATION)
        except Errors.KafkaError as err:
            log.error('Error sending %s to node %s [%s] -- marking coordinator dead', request.__class__.__name__, node_id, err)
            self.coordinator_dead()
            raise err
        return resp

    def check_errors(self):
        """ Check if coordinator is well and no authorization or unrecoverable
        errors occurred
        """
        if self._coordination_task.done():
            self._coordination_task.result()
        if self._error_consumed_fut is not None:
            self._error_consumed_fut.set_result(None)
            self._error_consumed_fut = None
        if self._pending_exception is not None:
            exc = self._pending_exception
            self._pending_exception = None
            raise exc

    def _push_error_to_user(self, exc):
        """ Most critical errors are not something we can continue execution
        without user action. Well right now we just drop the Consumer, but
        java client would certainly be ok if we just poll another time, maybe
        it will need to rejoin, but not fail with GroupAuthorizationFailedError
        till the end of days...
        XXX: Research if we can't have the same error several times. For
             example if user gets GroupAuthorizationFailedError and adds
             permission for the group, would Consumer work right away or would
             still raise exception a few times?
        """
        exc = copy.copy(exc)
        self._subscription.abort_waiters(exc)
        self._pending_exception = exc
        self._error_consumed_fut = create_future()
        return asyncio.wait([self._error_consumed_fut, self._closing], return_when=asyncio.FIRST_COMPLETED)

    async def close(self):
        """Close the coordinator, leave the current group
        and reset local generation/memberId."""
        if self._closing.done():
            return
        self._closing.set_result(None)
        if not self._coordination_task.done():
            await self._coordination_task
        await self._stop_heartbeat_task()
        await self._stop_commit_offsets_refresh_task()
        await self._maybe_leave_group()

    def maybe_leave_group(self):
        task = create_task(self._maybe_leave_group())
        return task

    async def _maybe_leave_group(self):
        if self.generation > 0 and self._group_instance_id is None:
            version = 0 if self._client.api_version < (0, 11, 0) else 1
            request = LeaveGroupRequest[version](self.group_id, self.member_id)
            try:
                await self._send_req(request)
            except Errors.KafkaError as err:
                log.error('LeaveGroup request failed: %s', err)
            else:
                log.info('LeaveGroup request succeeded')
        self.reset_generation()

    def _lookup_assignor(self, name):
        for assignor in self._assignors:
            if assignor.name == name:
                return assignor
        return None

    async def _on_join_prepare(self, previous_assignment):
        self._subscription.begin_reassignment()
        self._group_subscription = None
        if previous_assignment is not None:
            try:
                await self._maybe_do_last_autocommit(previous_assignment)
            except Errors.KafkaError as err:
                log.error('OffsetCommit failed before join, ignoring: %s', err)
            revoked = previous_assignment.tps
        else:
            revoked = set()
        log.info('Revoking previously assigned partitions %s for group %s', revoked, self.group_id)
        if self._subscription.listener:
            try:
                res = self._subscription.listener.on_partitions_revoked(revoked)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                log.exception('User provided subscription listener %s for group %s failed on_partitions_revoked', self._subscription.listener, self.group_id)

    async def _perform_assignment(self, response: Response):
        assignment_strategy = response.group_protocol
        members = response.members
        assignor = self._lookup_assignor(assignment_strategy)
        assert assignor, 'Invalid assignment protocol: %s' % assignment_strategy
        member_metadata = {}
        all_subscribed_topics = set()
        for member in members:
            if isinstance(response, JoinGroupResponse_v5):
                member_id, group_instance_id, metadata_bytes = member
            elif isinstance(response, (JoinGroupResponse[0], JoinGroupResponse[1], JoinGroupResponse[2])):
                member_id, metadata_bytes = member
            else:
                raise Exception('unknown protocol returned from assignment')
            metadata = ConsumerProtocol.METADATA.decode(metadata_bytes)
            member_metadata[member_id] = metadata
            all_subscribed_topics.update(metadata.subscription)
        self._group_subscription = all_subscribed_topics
        if not self._subscription.subscribed_pattern:
            self._client.set_topics(self._group_subscription)
        await self._client._maybe_wait_metadata()
        log.debug('Performing assignment for group %s using strategy %s with subscriptions %s', self.group_id, assignor.name, member_metadata)
        assignments = assignor.assign(self._cluster, member_metadata)
        log.debug('Finished assignment for group %s: %s', self.group_id, assignments)
        self._metadata_snapshot = self._get_metadata_snapshot()
        group_assignment = {}
        for member_id, assignment in assignments.items():
            group_assignment[member_id] = assignment
        return group_assignment

    async def _on_join_complete(self, generation, member_id, protocol, member_assignment_bytes):
        assignor = self._lookup_assignor(protocol)
        assert assignor, 'invalid assignment protocol: %s' % protocol
        assignment = ConsumerProtocol.ASSIGNMENT.decode(member_assignment_bytes)
        self._subscription.assign_from_subscribed(assignment.partitions())
        subscription = self._subscription.subscription
        assignor.on_assignment(assignment)
        await self._stop_commit_offsets_refresh_task()
        self.start_commit_offsets_refresh_task(subscription.assignment)
        assigned = set(self._subscription.assigned_partitions())
        log.info('Setting newly assigned partitions %s for group %s', assigned, self.group_id)
        if self._subscription.listener:
            try:
                res = self._subscription.listener.on_partitions_assigned(assigned)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                log.exception('User provided listener %s for group %s failed on partition assignment: %s', self._subscription.listener, self.group_id, assigned)

    def coordinator_dead(self):
        """ Mark the current coordinator as dead.
        NOTE: this will not force a group rejoin. If new coordinator is able to
        recognize this member we will just continue with current generation.
        """
        if self.coordinator_id is not None:
            log.warning('Marking the coordinator dead (node %s)for group %s.', self.coordinator_id, self.group_id)
            self.coordinator_id = None
            self._coordinator_dead_fut.set_result(None)

    def reset_generation(self):
        """ Coordinator did not recognize either generation or member_id. Will
        need to re-join the group.
        """
        self.generation = OffsetCommitRequest.DEFAULT_GENERATION_ID
        self.member_id = JoinGroupRequest[0].UNKNOWN_MEMBER_ID
        self.request_rejoin()

    def request_rejoin(self):
        if not self._rejoin_needed_fut.done():
            self._rejoin_needed_fut.set_result(None)

    def need_rejoin(self, subscription):
        """Check whether the group should be rejoined

        Returns:
            bool: True if consumer should rejoin group, False otherwise
        """
        return subscription.assignment is None or self._rejoin_needed_fut.done()

    async def ensure_coordinator_known(self):
        """ Block until the coordinator for this group is known.
        """
        if self.coordinator_id is not None:
            return
        async with self._coordinator_lookup_lock:
            retry_backoff = self._retry_backoff_ms / 1000
            while self.coordinator_id is None and (not self._closing.done()):
                try:
                    coordinator_id = await self._client.coordinator_lookup(CoordinationType.GROUP, self.group_id)
                except Errors.GroupAuthorizationFailedError:
                    err = Errors.GroupAuthorizationFailedError(self.group_id)
                    raise err
                except Errors.KafkaError as err:
                    log.error('Group Coordinator Request failed: %s', err)
                    if err.retriable:
                        await self._client.force_metadata_update()
                        await asyncio.sleep(retry_backoff)
                        continue
                    else:
                        raise
                ready = await self._client.ready(coordinator_id, group=ConnectionGroup.COORDINATION)
                if not ready:
                    await asyncio.sleep(retry_backoff)
                    continue
                self.coordinator_id = coordinator_id
                self._coordinator_dead_fut = create_future()
                log.info('Discovered coordinator %s for group %s', self.coordinator_id, self.group_id)

    async def _coordination_routine(self):
        try:
            await self.__coordination_routine()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error('Unexpected error in coordinator routine', exc_info=True)
            kafka_exc = Errors.KafkaError(f'Unexpected error during coordination {exc!r}')
            self._subscription.abort_waiters(kafka_exc)
            raise kafka_exc

    async def __coordination_routine(self):
        """ Main background task, that keeps track of changes in group
        coordination. This task will spawn/stop heartbeat task and perform
        autocommit in times it's safe to do so.
        """
        subscription = self._subscription.subscription
        assignment = None
        while not self._closing.done():
            if subscription is not None and (not subscription.active):
                self.request_rejoin()
                subscription = self._subscription.subscription
            if subscription is None:
                await asyncio.wait([self._subscription.wait_for_subscription(), self._closing], return_when=asyncio.FIRST_COMPLETED)
                if self._closing.done():
                    break
                subscription = self._subscription.subscription
            assert subscription is not None and subscription.active
            auto_assigned = self._subscription.partitions_auto_assigned()
            try:
                await self.ensure_coordinator_known()
                if auto_assigned and self.need_rejoin(subscription):
                    new_assignment = await self.ensure_active_group(subscription, assignment)
                    if new_assignment is None or not new_assignment.active:
                        continue
                    else:
                        assignment = new_assignment
                else:
                    assignment = subscription.assignment
                assert assignment is not None and assignment.active
                wait_timeout = await self._maybe_do_autocommit(assignment)
            except Errors.KafkaError as exc:
                await self._push_error_to_user(exc)
                continue
            futures = [self._closing, self._coordinator_dead_fut, subscription.unsubscribe_future]
            if auto_assigned:
                futures.append(self._rejoin_needed_fut)
            if self._heartbeat_task:
                futures.append(self._heartbeat_task)
            if self._commit_refresh_task:
                futures.append(self._commit_refresh_task)
            done, _ = await asyncio.wait(futures, timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED)
            for task in [self._heartbeat_task, self._commit_refresh_task]:
                if task and task.done():
                    exc = task.exception()
                    if exc:
                        await self._push_error_to_user(exc)
        if assignment is not None:
            try:
                await self._maybe_do_last_autocommit(assignment)
            except Errors.KafkaError as err:
                log.error('Failed to commit on finallization: %s', err)

    async def ensure_active_group(self, subscription, prev_assignment):
        if self._subscription.subscribed_pattern:
            await self._client.force_metadata_update()
            if not subscription.active:
                return None
        if not self._performed_join_prepare:
            await self._on_join_prepare(prev_assignment)
            self._performed_join_prepare = True
        await self._stop_heartbeat_task()
        idle_time = self._subscription.fetcher_idle_time
        if prev_assignment is not None and idle_time >= self._max_poll_interval:
            await asyncio.sleep(self._retry_backoff_ms / 1000)
            return None
        success = await self._do_rejoin_group(subscription)
        if success:
            self._performed_join_prepare = False
            self._start_heartbeat_task()
            return subscription.assignment
        return None

    def _start_heartbeat_task(self):
        if self._heartbeat_task is None:
            self._heartbeat_task = create_task(self._heartbeat_routine())

    async def _stop_heartbeat_task(self):
        if self._heartbeat_task is not None:
            if not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                await self._heartbeat_task
            self._heartbeat_task = None

    async def _heartbeat_routine(self):
        last_ok_heartbeat = time.monotonic()
        hb_interval = self._heartbeat_interval_ms / 1000
        session_timeout = self._session_timeout_ms / 1000
        retry_backoff = self._retry_backoff_ms / 1000
        sleep_time = hb_interval
        while self.member_id != JoinGroupRequest[0].UNKNOWN_MEMBER_ID:
            try:
                await asyncio.sleep(sleep_time)
                await self.ensure_coordinator_known()
                t0 = time.monotonic()
                success = await self._do_heartbeat()
            except asyncio.CancelledError:
                break
            if success:
                last_ok_heartbeat = time.monotonic()
                sleep_time = max((0, hb_interval - last_ok_heartbeat + t0))
            else:
                sleep_time = retry_backoff
            session_time = time.monotonic() - last_ok_heartbeat
            if session_time > session_timeout:
                log.error('Heartbeat session expired - marking coordinator dead')
                self.coordinator_dead()
            idle_time = self._subscription.fetcher_idle_time
            if idle_time < self._max_poll_interval:
                sleep_time = min(sleep_time, self._max_poll_interval - idle_time)
            else:
                await self._maybe_leave_group()
        log.debug('Stopping heartbeat task')

    async def _do_heartbeat(self):
        version = 0 if self._client.api_version < (0, 11, 0) else 1
        request = HeartbeatRequest[version](self.group_id, self.generation, self.member_id)
        log.debug('Heartbeat: %s[%s] %s', self.group_id, self.generation, self.member_id)
        try:
            resp = await self._send_req(request)
        except Errors.KafkaError as err:
            log.error('Heartbeat send request failed: %s. Will retry.', err)
            return False
        error_type = Errors.for_code(resp.error_code)
        if error_type is Errors.NoError:
            log.debug('Received successful heartbeat response for group %s', self.group_id)
            return True
        if error_type in (Errors.GroupCoordinatorNotAvailableError, Errors.NotCoordinatorForGroupError):
            log.warning('Heartbeat failed for group %s: coordinator (node %s) is either not started or not valid', self.group_id, self.coordinator_id)
            self.coordinator_dead()
        elif error_type is Errors.RebalanceInProgressError:
            log.warning('Heartbeat failed for group %s because it is rebalancing', self.group_id)
            self.request_rejoin()
            return True
        elif error_type is Errors.IllegalGenerationError:
            log.warning('Heartbeat failed for group %s: generation id is not  current.', self.group_id)
            self.reset_generation()
        elif error_type is Errors.UnknownMemberIdError:
            log.warning('Heartbeat failed: local member_id was not recognized; resetting and re-joining group')
            self.reset_generation()
        elif error_type is Errors.GroupAuthorizationFailedError:
            raise error_type(self.group_id)
        else:
            err = Errors.KafkaError(f'Unexpected exception in heartbeat task: {error_type()!r}')
            log.error('Heartbeat failed: %r', err)
            raise err
        return False

    def start_commit_offsets_refresh_task(self, assignment):
        if self._commit_refresh_task is not None:
            self._commit_refresh_task.cancel()
        self._commit_refresh_task = create_task(self._commit_refresh_routine(assignment))

    async def _stop_commit_offsets_refresh_task(self):
        if self._commit_refresh_task is not None:
            if not self._commit_refresh_task.done():
                self._commit_refresh_task.cancel()
                await self._commit_refresh_task
            self._commit_refresh_task = None

    async def _commit_refresh_routine(self, assignment):
        """ Task that will do a commit cache refresh if someone is waiting for
        it.
        """
        retry_backoff_ms = self._retry_backoff_ms / 1000
        commit_refresh_needed = assignment.commit_refresh_needed
        event_waiter = None
        try:
            while assignment.active:
                commit_refresh_needed.clear()
                success = await self._maybe_refresh_commit_offsets(assignment)
                wait_futures = [assignment.unassign_future]
                if not success:
                    timeout = retry_backoff_ms
                else:
                    timeout = None
                    event_waiter = create_task(commit_refresh_needed.wait())
                    wait_futures.append(event_waiter)
                await asyncio.wait(wait_futures, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            pass
        except Exception:
            commit_refresh_needed.set()
            raise
        if event_waiter is not None and (not event_waiter.done()):
            event_waiter.cancel()
            event_waiter = None

    async def _do_rejoin_group(self, subscription):
        rebalance = CoordinatorGroupRebalance(self, self.group_id, self.coordinator_id, subscription, self._assignors, self._session_timeout_ms, self._retry_backoff_ms)
        assignment = await rebalance.perform_group_join()
        if not subscription.active:
            log.debug('Subscription changed during rebalance from %s to %s. Rejoining group.', subscription.topics, self._subscription.topics)
            return False
        if assignment is None:
            await asyncio.sleep(self._retry_backoff_ms / 1000)
            return False
        protocol, member_assignment_bytes = assignment
        await self._on_join_complete(self.generation, self.member_id, protocol, member_assignment_bytes)
        return True

    async def _maybe_do_autocommit(self, assignment):
        if not self._enable_auto_commit:
            return None
        now = time.monotonic()
        interval = self._auto_commit_interval_ms / 1000
        backoff = self._retry_backoff_ms / 1000
        if now > self._next_autocommit_deadline:
            try:
                async with self._commit_lock:
                    await self._do_commit_offsets(assignment, assignment.all_consumed_offsets())
            except Errors.KafkaError as error:
                log.warning('Auto offset commit failed: %s', error)
                if self._is_commit_retriable(error):
                    self._next_autocommit_deadline = time.monotonic() + backoff
                    return backoff
                else:
                    raise
            self._next_autocommit_deadline = now + interval
        return max(0, self._next_autocommit_deadline - time.monotonic())

    def _is_commit_retriable(self, error):
        return error.retriable or isinstance(error, (Errors.UnknownMemberIdError, Errors.IllegalGenerationError, Errors.RebalanceInProgressError))

    async def _maybe_do_last_autocommit(self, assignment):
        if not self._enable_auto_commit:
            return
        await self.commit_offsets(assignment, assignment.all_consumed_offsets())

    async def commit_offsets(self, assignment, offsets):
        """Commit specific offsets

        Arguments:
            offsets (dict {TopicPartition: OffsetAndMetadata}): what to commit

        Raises KafkaError on failure
        """
        while True:
            await self.ensure_coordinator_known()
            try:
                async with self._commit_lock:
                    await asyncio.shield(self._do_commit_offsets(assignment, offsets))
            except (Errors.UnknownMemberIdError, Errors.IllegalGenerationError, Errors.RebalanceInProgressError):
                raise Errors.CommitFailedError('Commit cannot be completed since the group has already rebalanced and may have assigned the partitions to another member')
            except Errors.KafkaError as err:
                if not err.retriable:
                    raise err
                else:
                    await asyncio.sleep(self._retry_backoff_ms / 1000)
            else:
                break

    async def _do_commit_offsets(self, assignment, offsets):
        if not offsets:
            return
        offset_data = collections.defaultdict(list)
        for tp, offset in offsets.items():
            offset_data[tp.topic].append((tp.partition, offset.offset, offset.metadata))
        request = OffsetCommitRequest(self.group_id, self.generation, self.member_id, OffsetCommitRequest.DEFAULT_RETENTION_TIME, [(topic, tp_offsets) for topic, tp_offsets in offset_data.items()])
        log.debug('Sending offset-commit request with %s for group %s to %s', offsets, self.group_id, self.coordinator_id)
        response = await self._send_req(request)
        errored = collections.OrderedDict()
        unauthorized_topics = set()
        for topic, partitions in response.topics:
            for partition, error_code in partitions:
                tp = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                offset = offsets[tp]
                if error_type is Errors.NoError:
                    log.debug('Committed offset %s for partition %s', offset, tp)
                elif error_type is Errors.GroupAuthorizationFailedError:
                    log.error('OffsetCommit failed for group %s - %s', self.group_id, error_type.__name__)
                    errored[tp] = error_type(self.group_id)
                elif error_type is Errors.TopicAuthorizationFailedError:
                    unauthorized_topics.add(topic)
                elif error_type in (Errors.OffsetMetadataTooLargeError, Errors.InvalidCommitOffsetSizeError):
                    log.info('OffsetCommit failed for group %s on partition %s due to %s, will retry', self.group_id, tp, error_type.__name__)
                    errored[tp] = error_type()
                elif error_type is Errors.GroupLoadInProgressError:
                    log.info('OffsetCommit failed for group %s because group is initializing (%s), will retry', self.group_id, error_type.__name__)
                    errored[tp] = error_type()
                elif error_type in (Errors.GroupCoordinatorNotAvailableError, Errors.NotCoordinatorForGroupError, Errors.RequestTimedOutError):
                    log.info('OffsetCommit failed for group %s due to a coordinator error (%s), will find new coordinator and retry', self.group_id, error_type.__name__)
                    self.coordinator_dead()
                    errored[tp] = error_type()
                elif error_type in (Errors.UnknownMemberIdError, Errors.IllegalGenerationError, Errors.RebalanceInProgressError):
                    error = error_type(self.group_id)
                    log.error('OffsetCommit failed for group %s due to group error (%s), will rejoin', self.group_id, error)
                    if error_type is Errors.RebalanceInProgressError:
                        self.request_rejoin()
                    else:
                        self.reset_generation()
                    error = error_type(self.group_id)
                    log.error('OffsetCommit failed for group %s due to group error (%s), will rejoin', self.group_id, error)
                    errored[tp] = error
                else:
                    log.error('OffsetCommit failed for group %s on partition %s with offset %s: %s', self.group_id, tp, offset, error_type.__name__)
                    errored[tp] = error_type()
        if errored:
            first_error = list(errored.values())[0]
            raise first_error
        if unauthorized_topics:
            log.error('OffsetCommit failed for unauthorized topics %s', unauthorized_topics)
            raise Errors.TopicAuthorizationFailedError(unauthorized_topics)

    async def _maybe_refresh_commit_offsets(self, assignment):
        need_update = assignment.requesting_committed()
        if need_update:
            try:
                offsets = await self._do_fetch_commit_offsets(need_update)
            except Errors.KafkaError as err:
                if not err.retriable:
                    raise
                else:
                    log.debug('Failed to fetch committed offsets: %r', err)
                return False
            for tp in need_update:
                tp_state = assignment.state_value(tp)
                if tp in offsets:
                    tp_state.update_committed(offsets[tp])
                else:
                    tp_state.update_committed(OffsetAndMetadata(UNKNOWN_OFFSET, ''))
        return True

    async def fetch_committed_offsets(self, partitions):
        """Fetch the current committed offsets for specified partitions

        Arguments:
            partitions (list of TopicPartition): partitions to fetch

        Returns:
            dict: {TopicPartition: OffsetAndMetadata}
        """
        if not partitions:
            return {}
        while True:
            await self.ensure_coordinator_known()
            try:
                offsets = await self._do_fetch_commit_offsets(partitions)
            except Errors.KafkaError as err:
                if not err.retriable:
                    raise err
                else:
                    await asyncio.sleep(self._retry_backoff_ms / 1000)
            else:
                return offsets

    async def _do_fetch_commit_offsets(self, partitions):
        log.debug('Fetching committed offsets for partitions: %s', partitions)
        topic_partitions = collections.defaultdict(list)
        for tp in partitions:
            topic_partitions[tp.topic].append(tp.partition)
        request = OffsetFetchRequest(self.group_id, list(topic_partitions.items()))
        response = await self._send_req(request)
        offsets = {}
        for topic, partitions in response.topics:
            for partition, offset, metadata, error_code in partitions:
                tp = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                if error_type is not Errors.NoError:
                    error = error_type()
                    log.debug('Error fetching offset for %s: %s', tp, error)
                    if error_type is Errors.GroupLoadInProgressError:
                        raise error
                    elif error_type is Errors.NotCoordinatorForGroupError:
                        self.coordinator_dead()
                        raise error
                    elif error_type is Errors.UnknownTopicOrPartitionError:
                        log.warning('OffsetFetchRequest -- unknown topic %s', topic)
                        continue
                    elif error_type is Errors.GroupAuthorizationFailedError:
                        raise error_type(self.group_id)
                    else:
                        log.error('Unknown error fetching offsets for %s: %s', tp, error)
                        raise Errors.KafkaError(repr(error))
                if offset == UNKNOWN_OFFSET:
                    log.debug('No committed offset for partition %s', tp)
                else:
                    offsets[tp] = OffsetAndMetadata(offset, metadata)
        return offsets