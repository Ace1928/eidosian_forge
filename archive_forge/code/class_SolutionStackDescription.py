from datetime import datetime
from boto.compat import six
class SolutionStackDescription(BaseObject):

    def __init__(self, response):
        super(SolutionStackDescription, self).__init__()
        self.permitted_file_types = []
        if response['PermittedFileTypes']:
            for member in response['PermittedFileTypes']:
                permitted_file_type = str(member)
                self.permitted_file_types.append(permitted_file_type)
        self.solution_stack_name = str(response['SolutionStackName'])