class PercentAssignmentsApprovedRequirement(Requirement):
    """
    The percentage of assignments the Worker has submitted that were subsequently approved by the Requester, over all assignments the Worker has submitted. The value is an integer between 0 and 100.
    """

    def __init__(self, comparator, integer_value, required_to_preview=False):
        super(PercentAssignmentsApprovedRequirement, self).__init__(qualification_type_id='000000000000000000L0', comparator=comparator, integer_value=integer_value, required_to_preview=required_to_preview)