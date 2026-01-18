import logging
import threading
from typing import Dict, Optional
from wandb.proto.wandb_internal_pb2 import Record, Result
def add_from_record(self, record: Record) -> Optional[Context]:
    context_id = context_id_from_record(record)
    if not context_id:
        return None
    context_obj = self.add(context_id)
    return context_obj