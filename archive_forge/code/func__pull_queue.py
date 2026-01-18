import asyncio
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, Optional, Type, Union
import requests
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
def _pull_queue(self) -> None:
    try:
        from nucliadb_protos.writer_pb2 import BrokerMessage
    except ImportError as e:
        raise ImportError('nucliadb-protos is not installed. Run `pip install nucliadb-protos` to install.') from e
    try:
        from google.protobuf.json_format import MessageToJson
    except ImportError as e:
        raise ImportError('Unable to import google.protobuf, please install with `pip install protobuf`.') from e
    res = requests.get(self._config['BACKEND'] + '/processing/pull', headers={'x-stf-nuakey': 'Bearer ' + self._config['NUA_KEY']}).json()
    if res['status'] == 'empty':
        logger.info('Queue empty')
    elif res['status'] == 'ok':
        payload = res['payload']
        pb = BrokerMessage()
        pb.ParseFromString(base64.b64decode(payload))
        uuid = pb.uuid
        logger.info(f'Pulled {uuid} from queue')
        matching_id = self._find_matching_id(uuid)
        if not matching_id:
            logger.info(f'No matching id for {uuid}')
        else:
            self._results[matching_id]['status'] = 'done'
            data = MessageToJson(pb, preserving_proto_field_name=True, including_default_value_fields=True)
            self._results[matching_id]['data'] = data