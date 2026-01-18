from osc_lib.command import command
from osc_lib import exceptions
class TroveDeleter(command.Command):

    def delete_resources(self, ids):
        """Delete one or more resources."""
        failure_flag = False
        success_msg = 'Request to delete %s %s has been accepted.'
        error_msg = 'Unable to delete the specified %s(s).'
        for id in ids:
            try:
                self.delete_func(id)
                print(success_msg % (self.resource, id))
            except Exception as e:
                failure_flag = True
                print(e)
        if failure_flag:
            raise exceptions.CommandError(error_msg % self.resource)