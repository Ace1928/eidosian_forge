import copy
import math
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import ChunksizeAdjuster
def _extra_upload_part_args(self, extra_args):
    return get_filtered_dict(extra_args, self.UPLOAD_PART_COPY_ARGS)