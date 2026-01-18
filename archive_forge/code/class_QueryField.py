import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class QueryField(ReportAPIBaseModel):
    args: LList[QueryFieldsValue] = Field(default_factory=lambda: [QueryFieldsValue(name='runSets', value='${runSets}'), QueryFieldsValue(name='limit', value=500)])
    fields: LList[QueryFieldsField] = Field(default_factory=lambda: [QueryFieldsField(name='id', value=[], fields=None), QueryFieldsField(name='name', value=[], fields=None)])
    name: str = 'runSets'