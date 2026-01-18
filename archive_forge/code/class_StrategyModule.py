from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError, AnsibleParserError
from ansible.executor.play_iterator import IteratingStates, FailedStates
from ansible.module_utils.common.text.converters import to_text
from ansible.playbook.handler import Handler
from ansible.playbook.included_file import IncludedFile
from ansible.playbook.task import Task
from ansible.plugins.loader import action_loader
from ansible.plugins.strategy import StrategyBase
from ansible.template import Templar
from ansible.utils.display import Display
class StrategyModule(StrategyBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_handlers = False

    def _get_next_task_lockstep(self, hosts, iterator):
        """
        Returns a list of (host, task) tuples, where the task may
        be a noop task to keep the iterator in lock step across
        all hosts.
        """
        noop_task = Task()
        noop_task.action = 'meta'
        noop_task.args['_raw_params'] = 'noop'
        noop_task.implicit = True
        noop_task.set_loader(iterator._play._loader)
        state_task_per_host = {}
        for host in hosts:
            state, task = iterator.get_next_task_for_host(host, peek=True)
            if task is not None:
                state_task_per_host[host] = (state, task)
        if not state_task_per_host:
            return [(h, None) for h in hosts]
        if self._in_handlers and (not any(filter(lambda rs: rs == IteratingStates.HANDLERS, (s.run_state for s, dummy in state_task_per_host.values())))):
            self._in_handlers = False
        if self._in_handlers:
            lowest_cur_handler = min((s.cur_handlers_task for s, t in state_task_per_host.values() if s.run_state == IteratingStates.HANDLERS))
        else:
            task_uuids = [t._uuid for s, t in state_task_per_host.values()]
            _loop_cnt = 0
            while _loop_cnt <= 1:
                try:
                    cur_task = iterator.all_tasks[iterator.cur_task]
                except IndexError:
                    iterator.cur_task = 0
                    _loop_cnt += 1
                else:
                    iterator.cur_task += 1
                    if cur_task._uuid in task_uuids:
                        break
            else:
                raise AnsibleAssertionError('BUG: There seems to be a mismatch between tasks in PlayIterator and HostStates.')
        host_tasks = []
        for host, (state, task) in state_task_per_host.items():
            if self._in_handlers and lowest_cur_handler == state.cur_handlers_task or (not self._in_handlers and cur_task._uuid == task._uuid):
                iterator.set_state_for_host(host.name, state)
                host_tasks.append((host, task))
            else:
                host_tasks.append((host, noop_task))
        if not self._in_handlers and cur_task.action in C._ACTION_META and (cur_task.args.get('_raw_params') == 'flush_handlers'):
            self._in_handlers = True
        return host_tasks

    def run(self, iterator, play_context):
        """
        The linear strategy is simple - get the next task and queue
        it for all hosts, then wait for the queue to drain before
        moving on to the next task
        """
        result = self._tqm.RUN_OK
        work_to_do = True
        self._set_hosts_cache(iterator._play)
        while work_to_do and (not self._tqm._terminated):
            try:
                display.debug('getting the remaining hosts for this loop')
                hosts_left = self.get_hosts_left(iterator)
                display.debug('done getting the remaining hosts for this loop')
                callback_sent = False
                work_to_do = False
                host_tasks = self._get_next_task_lockstep(hosts_left, iterator)
                skip_rest = False
                choose_step = True
                any_errors_fatal = False
                results = []
                for host, task in host_tasks:
                    if not task:
                        continue
                    if self._tqm._terminated:
                        break
                    run_once = False
                    work_to_do = True
                    if not isinstance(task, Handler) and task._role:
                        role_obj = self._get_cached_role(task, iterator._play)
                        if role_obj.has_run(host) and task._role._metadata.allow_duplicates is False:
                            display.debug("'%s' skipped because role has already run" % task)
                            continue
                    display.debug('getting variables')
                    task_vars = self._variable_manager.get_vars(play=iterator._play, host=host, task=task, _hosts=self._hosts_cache, _hosts_all=self._hosts_cache_all)
                    self.add_tqm_variables(task_vars, play=iterator._play)
                    templar = Templar(loader=self._loader, variables=task_vars)
                    display.debug('done getting variables')
                    task_action = templar.template(task.action)
                    try:
                        action = action_loader.get(task_action, class_only=True, collection_list=task.collections)
                    except KeyError:
                        action = None
                    if task_action in C._ACTION_META:
                        results.extend(self._execute_meta(task, play_context, iterator, host))
                        if task.args.get('_raw_params', None) not in ('noop', 'reset_connection', 'end_host', 'role_complete', 'flush_handlers'):
                            run_once = True
                        if (task.any_errors_fatal or run_once) and (not task.ignore_errors):
                            any_errors_fatal = True
                    else:
                        if self._step and choose_step:
                            if self._take_step(task):
                                choose_step = False
                            else:
                                skip_rest = True
                                break
                        run_once = templar.template(task.run_once) or (action and getattr(action, 'BYPASS_HOST_LOOP', False))
                        if (task.any_errors_fatal or run_once) and (not task.ignore_errors):
                            any_errors_fatal = True
                        if not callback_sent:
                            display.debug('sending task start callback, copying the task so we can template it temporarily')
                            saved_name = task.name
                            display.debug('done copying, going to template now')
                            try:
                                task.name = to_text(templar.template(task.name, fail_on_undefined=False), nonstring='empty')
                                display.debug('done templating')
                            except Exception:
                                display.debug('templating failed for some reason')
                            display.debug('here goes the callback...')
                            if isinstance(task, Handler):
                                self._tqm.send_callback('v2_playbook_on_handler_task_start', task)
                            else:
                                self._tqm.send_callback('v2_playbook_on_task_start', task, is_conditional=False)
                            task.name = saved_name
                            callback_sent = True
                            display.debug('sending task start callback')
                        self._blocked_hosts[host.get_name()] = True
                        self._queue_task(host, task, task_vars, play_context)
                        del task_vars
                    if isinstance(task, Handler):
                        if run_once:
                            task.clear_hosts()
                        else:
                            task.remove_host(host)
                    if run_once:
                        break
                    results.extend(self._process_pending_results(iterator, max_passes=max(1, int(len(self._tqm._workers) * 0.1))))
                if skip_rest:
                    continue
                display.debug('done queuing things up, now waiting for results queue to drain')
                if self._pending_results > 0:
                    results.extend(self._wait_on_pending_results(iterator))
                self.update_active_connections(results)
                included_files = IncludedFile.process_include_results(results, iterator=iterator, loader=self._loader, variable_manager=self._variable_manager)
                if len(included_files) > 0:
                    display.debug('we have included files to process')
                    display.debug('generating all_blocks data')
                    all_blocks = dict(((host, []) for host in hosts_left))
                    display.debug('done generating all_blocks data')
                    included_tasks = []
                    failed_includes_hosts = set()
                    for included_file in included_files:
                        display.debug('processing included file: %s' % included_file._filename)
                        is_handler = False
                        try:
                            if included_file._is_role:
                                new_ir = self._copy_included_file(included_file)
                                new_blocks, handler_blocks = new_ir.get_block_list(play=iterator._play, variable_manager=self._variable_manager, loader=self._loader)
                            else:
                                is_handler = isinstance(included_file._task, Handler)
                                new_blocks = self._load_included_file(included_file, iterator=iterator, is_handler=is_handler)
                            iterator.handlers = [h for b in iterator._play.handlers for h in b.block]
                            display.debug('iterating over new_blocks loaded from include file')
                            for new_block in new_blocks:
                                if is_handler:
                                    for task in new_block.block:
                                        task.notified_hosts = included_file._hosts[:]
                                    final_block = new_block
                                else:
                                    task_vars = self._variable_manager.get_vars(play=iterator._play, task=new_block.get_first_parent_include(), _hosts=self._hosts_cache, _hosts_all=self._hosts_cache_all)
                                    display.debug('filtering new block on tags')
                                    final_block = new_block.filter_tagged_tasks(task_vars)
                                    display.debug('done filtering new block on tags')
                                    included_tasks.extend(final_block.get_tasks())
                                for host in hosts_left:
                                    if host in included_file._hosts:
                                        all_blocks[host].append(final_block)
                            display.debug('done iterating over new_blocks loaded from include file')
                        except AnsibleParserError:
                            raise
                        except AnsibleError as e:
                            if included_file._is_role:
                                display.error(to_text(e), wrap_text=False)
                            for r in included_file._results:
                                r._result['failed'] = True
                                failed_includes_hosts.add(r._host)
                            continue
                    for host in failed_includes_hosts:
                        self._tqm._failed_hosts[host.name] = True
                        iterator.mark_host_failed(host)
                    display.debug('extending task lists for all hosts with included blocks')
                    for host in hosts_left:
                        iterator.add_tasks(host, all_blocks[host])
                    iterator.all_tasks[iterator.cur_task:iterator.cur_task] = included_tasks
                    display.debug('done extending task lists')
                    display.debug('done processing included files')
                display.debug('results queue empty')
                display.debug('checking for any_errors_fatal')
                failed_hosts = []
                unreachable_hosts = []
                for res in results:
                    if (res.is_failed() or res._task.action in C._ACTION_META) and iterator.is_failed(res._host):
                        failed_hosts.append(res._host.name)
                    elif res.is_unreachable():
                        unreachable_hosts.append(res._host.name)
                if any_errors_fatal and (len(failed_hosts) > 0 or len(unreachable_hosts) > 0):
                    dont_fail_states = frozenset([IteratingStates.RESCUE, IteratingStates.ALWAYS])
                    for host in hosts_left:
                        s, dummy = iterator.get_next_task_for_host(host, peek=True)
                        s = iterator.get_active_state(s)
                        if s.run_state not in dont_fail_states or (s.run_state == IteratingStates.RESCUE and s.fail_state & FailedStates.RESCUE != 0):
                            self._tqm._failed_hosts[host.name] = True
                            result |= self._tqm.RUN_FAILED_BREAK_PLAY
                display.debug('done checking for any_errors_fatal')
                display.debug('checking for max_fail_percentage')
                if iterator._play.max_fail_percentage is not None and len(results) > 0:
                    percentage = iterator._play.max_fail_percentage / 100.0
                    if len(self._tqm._failed_hosts) / iterator.batch_size > percentage:
                        for host in hosts_left:
                            if host.name not in failed_hosts:
                                self._tqm._failed_hosts[host.name] = True
                                iterator.mark_host_failed(host)
                        self._tqm.send_callback('v2_playbook_on_no_hosts_remaining')
                        result |= self._tqm.RUN_FAILED_BREAK_PLAY
                    display.debug('(%s failed / %s total )> %s max fail' % (len(self._tqm._failed_hosts), iterator.batch_size, percentage))
                display.debug('done checking for max_fail_percentage')
                display.debug('checking to see if all hosts have failed and the running result is not ok')
                if result != self._tqm.RUN_OK and len(self._tqm._failed_hosts) >= len(hosts_left):
                    display.debug('^ not ok, so returning result now')
                    self._tqm.send_callback('v2_playbook_on_no_hosts_remaining')
                    return result
                display.debug('done checking to see if all hosts have failed')
            except (IOError, EOFError) as e:
                display.debug('got IOError/EOFError in task loop: %s' % e)
                return self._tqm.RUN_UNKNOWN_ERROR
        return super(StrategyModule, self).run(iterator, play_context, result)