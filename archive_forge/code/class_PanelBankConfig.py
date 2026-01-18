import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class PanelBankConfig(ReportAPIBaseModel):
    state: int = 0
    settings: PanelBankConfigSettings = Field(default_factory=PanelBankConfigSettings)
    sections: LList[PanelBankConfigSectionsItem] = Field(default_factory=lambda: [PanelBankConfigSectionsItem()])