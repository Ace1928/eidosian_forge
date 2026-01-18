from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from wasabi import msg
class ProjectConfigAssetGit(BaseModel):
    git: ProjectConfigAssetGitItem = Field(..., title='Git repo information')
    checksum: Optional[str] = Field(None, title='MD5 hash of file', regex='([a-fA-F\\d]{32})')
    description: Optional[StrictStr] = Field(None, title='Description of asset')