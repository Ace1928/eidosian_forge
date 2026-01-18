import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class Vega2(Panel):
    view_type: Literal['Vega2'] = 'Vega2'
    config: Vega2Config = Field(default_factory=Vega2Config)