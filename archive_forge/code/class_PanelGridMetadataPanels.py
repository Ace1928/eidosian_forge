import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class PanelGridMetadataPanels(ReportAPIBaseModel):
    views: dict = Field(default_factory=lambda: {'0': {'name': 'Panels', 'defaults': [], 'config': []}})
    tabs: list = Field(default_factory=lambda: ['0'])
    ref: Optional[Ref] = None