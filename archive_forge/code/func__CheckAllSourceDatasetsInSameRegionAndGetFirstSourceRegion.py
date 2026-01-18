from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import time
from typing import List, Optional, Tuple
from absl import flags
from clients import bigquery_client
from clients import client_dataset
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from utils import bq_error
from utils import bq_id_utils
def _CheckAllSourceDatasetsInSameRegionAndGetFirstSourceRegion(self, client: bigquery_client.BigqueryClient, source_references: List[bq_id_utils.ApiClientHelper.TableReference]) -> Tuple[bool, Optional[str]]:
    """Checks whether all source datasets are from same region.

    Args:
      client: Bigquery client
      source_references: Source reference

    Returns:
      true  - all source datasets are from the same region. Includes the
              scenario in which there is only one source dataset
      false - all source datasets are not from the same region.
    Raises:
      bq_error.BigqueryNotFoundError: If unable to compute the dataset
        region
    """
    all_source_datasets_in_same_region = True
    first_source_region = None
    for _, val in enumerate(source_references):
        source_dataset = val.GetDatasetReference()
        source_region = client_dataset.GetDatasetRegion(apiclient=client.apiclient, reference=source_dataset)
        if source_region is None:
            raise bq_error.BigqueryNotFoundError(self._DATASET_NOT_FOUND % (str(source_dataset),), {'reason': 'notFound'}, [])
        if first_source_region is None:
            first_source_region = source_region
        elif first_source_region != source_region:
            all_source_datasets_in_same_region = False
            break
    return (all_source_datasets_in_same_region, first_source_region)