import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from aiokafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from aiokafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from aiokafka.coordinator.assignors.sticky.sorted_set import SortedSet
from aiokafka.coordinator.protocol import (
from aiokafka.coordinator.protocol import Schema
from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import String, Array, Int32
from aiokafka.structs import TopicPartition
class StickyPartitionAssignor(AbstractPartitionAssignor):
    """
    https://cwiki.apache.org/confluence/display/KAFKA/KIP-54+-+Sticky+Partition+Assignment+Strategy

    The sticky assignor serves two purposes. First, it guarantees an assignment that is
    as balanced as possible, meaning either:
    - the numbers of topic partitions assigned to consumers differ by at most one; or
    - each consumer that has 2+ fewer topic partitions than some other consumer cannot
    get any of those topic partitions transferred to it.

    Second, it preserved as many existing assignment as possible when a reassignment
    occurs. This helps in saving some of the overhead processing when topic partitions
    move from one consumer to another.

    Starting fresh it would work by distributing the partitions over consumers as evenly
    as possible. Even though this may sound similar to how round robin assignor works,
    the second example below shows that it is not. During a reassignment it would
    perform the reassignment in such a way that in the new assignment
    - topic partitions are still distributed as evenly as possible, and
    - topic partitions stay with their previously assigned consumers as much as
    possible.

    The first goal above takes precedence over the second one.

    Example 1.
    Suppose there are three consumers C0, C1, C2,
    four topics t0, t1, t2, t3, and each topic has 2 partitions,
    resulting in partitions t0p0, t0p1, t1p0, t1p1, t2p0, t2p1, t3p0, t3p1.
    Each consumer is subscribed to all three topics.

    The assignment with both sticky and round robin assignors will be:
    - C0: [t0p0, t1p1, t3p0]
    - C1: [t0p1, t2p0, t3p1]
    - C2: [t1p0, t2p1]

    Now, let's assume C1 is removed and a reassignment is about to happen. The round
    robin assignor would produce:
    - C0: [t0p0, t1p0, t2p0, t3p0]
    - C2: [t0p1, t1p1, t2p1, t3p1]

    while the sticky assignor would result in:
    - C0 [t0p0, t1p1, t3p0, t2p0]
    - C2 [t1p0, t2p1, t0p1, t3p1]
    preserving all the previous assignments (unlike the round robin assignor).


    Example 2.
    There are three consumers C0, C1, C2,
    and three topics t0, t1, t2, with 1, 2, and 3 partitions respectively.
    Therefore, the partitions are t0p0, t1p0, t1p1, t2p0, t2p1, t2p2.
    C0 is subscribed to t0;
    C1 is subscribed to t0, t1;
    and C2 is subscribed to t0, t1, t2.

    The round robin assignor would come up with the following assignment:
    - C0 [t0p0]
    - C1 [t1p0]
    - C2 [t1p1, t2p0, t2p1, t2p2]

    which is not as balanced as the assignment suggested by sticky assignor:
    - C0 [t0p0]
    - C1 [t1p0, t1p1]
    - C2 [t2p0, t2p1, t2p2]

    Now, if consumer C0 is removed, these two assignors would produce the following
    assignments. Round Robin (preserves 3 partition assignments):
    - C1 [t0p0, t1p1]
    - C2 [t1p0, t2p0, t2p1, t2p2]

    Sticky (preserves 5 partition assignments):
    - C1 [t1p0, t1p1, t0p0]
    - C2 [t2p0, t2p1, t2p2]
    """
    DEFAULT_GENERATION_ID = -1
    name = 'sticky'
    version = 0
    member_assignment = None
    generation = DEFAULT_GENERATION_ID
    _latest_partition_movements = None

    @classmethod
    def assign(cls, cluster, members):
        """Performs group assignment given cluster metadata and member subscriptions

        Arguments:
            cluster (ClusterMetadata): cluster metadata
            members (dict of {member_id: MemberMetadata}): decoded metadata for each
            member in the group.

        Returns:
          dict: {member_id: MemberAssignment}
        """
        members_metadata = {}
        for consumer, member_metadata in members.items():
            members_metadata[consumer] = cls.parse_member_metadata(member_metadata)
        executor = StickyAssignmentExecutor(cluster, members_metadata)
        executor.perform_initial_assignment()
        executor.balance()
        cls._latest_partition_movements = executor.partition_movements
        assignment = {}
        for member_id in members:
            assignment[member_id] = ConsumerProtocolMemberAssignment(cls.version, sorted(executor.get_final_assignment(member_id)), b'')
        return assignment

    @classmethod
    def parse_member_metadata(cls, metadata):
        """
        Parses member metadata into a python object.
        This implementation only serializes and deserializes the
        StickyAssignorMemberMetadataV1 user data, since no StickyAssignor written in
        Python was deployed ever in the wild with version V0, meaning that there is no
        need to support backward compatibility with V0.

        Arguments:
          metadata (MemberMetadata): decoded metadata for a member of the group.

        Returns:
          parsed metadata (StickyAssignorMemberMetadataV1)
        """
        user_data = metadata.user_data
        if not user_data:
            return StickyAssignorMemberMetadataV1(partitions=[], generation=cls.DEFAULT_GENERATION_ID, subscription=metadata.subscription)
        try:
            decoded_user_data = StickyAssignorUserDataV1.decode(user_data)
        except Exception as e:
            log.error('Could not parse member data', e)
            return StickyAssignorMemberMetadataV1(partitions=[], generation=cls.DEFAULT_GENERATION_ID, subscription=metadata.subscription)
        member_partitions = []
        for topic, partitions in decoded_user_data.previous_assignment:
            member_partitions.extend([TopicPartition(topic, partition) for partition in partitions])
        return StickyAssignorMemberMetadataV1(partitions=member_partitions, generation=decoded_user_data.generation, subscription=metadata.subscription)

    @classmethod
    def metadata(cls, topics):
        return cls._metadata(topics, cls.member_assignment, cls.generation)

    @classmethod
    def _metadata(cls, topics, member_assignment_partitions, generation=-1):
        if member_assignment_partitions is None:
            log.debug('No member assignment available')
            user_data = b''
        else:
            log.debug('Member assignment is available, generating the metadata: generation {}'.format(cls.generation))
            partitions_by_topic = defaultdict(list)
            for topic_partition in member_assignment_partitions:
                partitions_by_topic[topic_partition.topic].append(topic_partition.partition)
            data = StickyAssignorUserDataV1(partitions_by_topic.items(), generation)
            user_data = data.encode()
        return ConsumerProtocolMemberMetadata(cls.version, list(topics), user_data)

    @classmethod
    def on_assignment(cls, assignment):
        """Callback that runs on each assignment. Updates assignor's state.

        Arguments:
          assignment: MemberAssignment
        """
        log.debug('On assignment: assignment={}'.format(assignment))
        cls.member_assignment = assignment.partitions()

    @classmethod
    def on_generation_assignment(cls, generation):
        """Callback that runs on each assignment. Updates assignor's generation id.

        Arguments:
          generation: generation id
        """
        log.debug('On generation assignment: generation={}'.format(generation))
        cls.generation = generation