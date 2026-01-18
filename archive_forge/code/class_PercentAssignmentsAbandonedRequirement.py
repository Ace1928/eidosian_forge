class PercentAssignmentsAbandonedRequirement(Requirement):
    """
    The percentage of assignments the Worker has abandoned (allowed the deadline to elapse), over all assignments the Worker has accepted. The value is an integer between 0 and 100.
    """

    def __init__(self, comparator, integer_value, required_to_preview=False):
        super(PercentAssignmentsAbandonedRequirement, self).__init__(qualification_type_id='00000000000000000070', comparator=comparator, integer_value=integer_value, required_to_preview=required_to_preview)