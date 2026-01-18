import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class PanelBankConfigSectionsItem(ReportAPIBaseModel):
    name: str = 'Hidden Panels'
    is_open: bool = False
    type: str = 'flow'
    flow_config: FlowConfig = Field(default_factory=FlowConfig)
    sorted: int = 0
    local_panel_settings: LocalPanelSettings = Field(default_factory=LocalPanelSettings)
    panels: list = Field(default_factory=list)
    ref: Optional[Ref] = None