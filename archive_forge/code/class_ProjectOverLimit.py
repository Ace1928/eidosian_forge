from oslo_limit._i18n import _
class ProjectOverLimit(Exception):

    def __init__(self, project_id, over_limit_info_list):
        """Exception raised when a project goes over one or more limits

        :param project_id: the project id
        :param over_limit_info_list: list of OverLimitInfo objects
        """
        if not isinstance(over_limit_info_list, list):
            raise ValueError(over_limit_info_list)
        if len(over_limit_info_list) == 0:
            raise ValueError(over_limit_info_list)
        for info in over_limit_info_list:
            if not isinstance(info, OverLimitInfo):
                raise ValueError(over_limit_info_list)
        self.project_id = project_id
        self.over_limit_info_list = over_limit_info_list
        msg = _('Project %(project_id)s is over a limit for %(limits)s') % {'project_id': project_id, 'limits': over_limit_info_list}
        super(ProjectOverLimit, self).__init__(msg)