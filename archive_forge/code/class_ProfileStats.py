import re
class ProfileStats:
    """
    ProfileStats, runtime execution statistics of operation.
    """

    def __init__(self, records_produced, execution_time):
        self.records_produced = records_produced
        self.execution_time = execution_time