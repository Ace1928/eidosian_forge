from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import progress_tracker
def _UploadSourceStage():
    return progress_tracker.Stage('Uploading sources...', key=UPLOAD_SOURCE)