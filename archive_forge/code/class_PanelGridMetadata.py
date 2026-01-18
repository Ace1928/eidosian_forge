import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
class PanelGridMetadata(ReportAPIBaseModel):
    open_viz: bool = True
    open_run_set: Optional[int] = 0
    name: Literal['unused-name'] = 'unused-name'
    run_sets: LList['Runset'] = Field(default_factory=lambda: [Runset()])
    panels: PanelGridMetadataPanels = Field(default_factory=PanelGridMetadataPanels)
    panel_bank_config: PanelBankConfig = Field(default_factory=PanelBankConfig)
    panel_bank_section_config: PanelBankSectionConfig = Field(default_factory=PanelBankSectionConfig)
    custom_run_colors: Dict[str, str] = Field(default_factory=dict)