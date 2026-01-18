from typing import Dict, List, NamedTuple, Optional
from googleapiclient import discovery
from clients import utils as bq_client_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
Deletes DatasetReference reference.

  Args:
    apiclient: the api client to make the request with.
    reference: the DatasetReference to delete.
    ignore_not_found: Whether to ignore "not found" errors.
    delete_contents: [Boolean] Whether to delete the contents of non-empty
      datasets. If not specified and the dataset has tables in it, the delete
      will fail. If not specified, the server default applies.

  Raises:
    TypeError: if reference is not a DatasetReference.
    bq_error.BigqueryNotFoundError: if reference does not exist and
      ignore_not_found is False.
  