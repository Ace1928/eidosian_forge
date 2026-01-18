import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def _load_legacy_config(config_file: str) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
    log.debug('Attempting to parse legacy auth file format')
    try:
        data = []
        with open(config_file) as f:
            for line in f.readlines():
                data.append(line.strip().split(' = ')[1])
            if len(data) < 2:
                raise InvalidConfigFileError('Invalid or empty configuration file!')
        username, password = decode_auth(data[0])
        return {'auths': {INDEX_NAME: {'username': username, 'password': password, 'email': data[1], 'serveraddress': INDEX_URL}}}
    except Exception as e:
        log.debug(e)
        pass
    log.debug('All parsing attempts failed - returning empty config')
    return {}