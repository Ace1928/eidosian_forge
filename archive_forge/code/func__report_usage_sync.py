import asyncio
import logging
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor
import ray
import ray._private.usage.usage_lib as ray_usage_lib
from ray._private.utils import get_or_create_event_loop
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.utils import async_loop_forever
def _report_usage_sync(self):
    """
        - Always write usage_stats.json regardless of report success/failure.
        - If report fails, the error message should be written to usage_stats.json
        - If file write fails, the error will just stay at dashboard.log.
            usage_stats.json won't be written.
        """
    if not self.usage_stats_enabled:
        return
    try:
        self._fetch_and_record_extra_usage_stats_data()
        data = ray_usage_lib.generate_report_data(self.cluster_config_to_report, self.total_success, self.total_failed, self.seq_no, self._dashboard_head.gcs_client.address)
        error = None
        try:
            self.client.report_usage_data(ray_usage_lib._usage_stats_report_url(), data)
        except Exception as e:
            logger.info(f'Usage report request failed. {e}')
            error = str(e)
            self.total_failed += 1
        else:
            self.total_success += 1
        finally:
            self.seq_no += 1
        data = ray_usage_lib.generate_write_data(data, error)
        self.client.write_usage_data(data, self.session_dir)
    except Exception as e:
        logger.exception(e)
        logger.info(f'Usage report failed: {e}')