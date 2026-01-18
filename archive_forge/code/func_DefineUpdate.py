from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def DefineUpdate(self, update_field_name):
    """Defines the Update functionality on the calling class.

    Args:
      update_field_name: the field on the patch_request to assign updated object
                         to
    """

    def Update(self, updating_object, update_mask=None):
        """Updates an object.

      Args:
        self: The self of the class this is set on.
        updating_object: Object which is being updated.
        update_mask: A string saying which fields have been updated.

      Returns:
        Long running operation.
      """
        req = self.patch_request(name=updating_object.name, updateMask=update_mask)
        setattr(req, update_field_name, updating_object)
        return self.service.Patch(req)
    setattr(self, 'Update', types.MethodType(Update, self))