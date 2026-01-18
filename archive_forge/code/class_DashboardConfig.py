from dataclasses import dataclass
from typing import List, Optional
@dataclass
class DashboardConfig:
    name: str
    default_uid: str
    panels: List[Panel]
    standard_global_filters: List[str]
    base_json_file_name: str