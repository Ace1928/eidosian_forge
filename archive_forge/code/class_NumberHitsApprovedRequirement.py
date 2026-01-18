class NumberHitsApprovedRequirement(Requirement):
    """
    Specifies the total number of HITs submitted by a Worker that have been approved. The value is an integer greater than or equal to 0.

    If specifying a Country and Subdivision, use a tuple of valid  ISO 3166 country code and ISO 3166-2 subdivision code, e.g. ('US', 'CA') for the US State of California.

    When using the 'In' and 'NotIn', locale should be a list of Countries and/or (Country, Subdivision) tuples.

    """

    def __init__(self, comparator, integer_value, required_to_preview=False):
        super(NumberHitsApprovedRequirement, self).__init__(qualification_type_id='00000000000000000040', comparator=comparator, integer_value=integer_value, required_to_preview=required_to_preview)