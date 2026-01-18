from __future__ import annotations
import json
import os
from glob import glob
from typing import Any
import json5
from jsonschema import Draft7Validator as Validator
from jsonschema import ValidationError
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import APIHandler
from jupyter_server.services.config.manager import ConfigManager, recursive_update
from tornado import web
from .translation_utils import DEFAULT_LOCALE, L10N_SCHEMA_NAME, SYS_LOCALE, is_valid_locale
def _get_user_settings(settings_dir: str, schema_name: str, schema: Any) -> dict[str, Any]:
    """
    Returns a dictionary containing the raw user settings, the parsed user
    settings, a validation warning for a schema, and file times.
    """
    path = _path(settings_dir, schema_name, False, SETTINGS_EXTENSION)
    raw = '{}'
    settings = {}
    warning = None
    validation_warning = 'Failed validating settings (%s): %s'
    parse_error = 'Failed loading settings (%s): %s'
    last_modified = None
    created = None
    if os.path.exists(path):
        stat = os.stat(path)
        last_modified = tz.utcfromtimestamp(stat.st_mtime).isoformat()
        created = tz.utcfromtimestamp(stat.st_ctime).isoformat()
        with open(path, encoding='utf-8') as fid:
            try:
                raw = fid.read() or raw
                settings = json5.loads(raw)
            except Exception as e:
                raise web.HTTPError(500, parse_error % (schema_name, str(e))) from None
    if len(settings):
        validator = Validator(schema)
        try:
            validator.validate(settings)
        except ValidationError as e:
            warning = validation_warning % (schema_name, str(e))
            raw = '{}'
            settings = {}
    return dict(raw=raw, settings=settings, warning=warning, last_modified=last_modified, created=created)