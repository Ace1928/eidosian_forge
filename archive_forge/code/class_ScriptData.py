from __future__ import annotations
import os
from dataclasses import dataclass, field
@dataclass(frozen=True)
class ScriptData:
    """Contains parameters related to running a script."""
    main_script_path: str
    is_hello: bool = False
    script_folder: str = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Set some computed values derived from main_script_path.

        The usage of object.__setattr__ is necessary because trying to set
        self.script_folder or self.name normally, even within the __init__ method, will
        explode since we declared this dataclass to be frozen.

        We do this in __post_init__ so that we can use the auto-generated __init__
        method that most dataclasses use.
        """
        main_script_path = os.path.abspath(self.main_script_path)
        script_folder = os.path.dirname(main_script_path)
        object.__setattr__(self, 'script_folder', script_folder)
        basename = os.path.basename(main_script_path)
        name = str(os.path.splitext(basename)[0])
        object.__setattr__(self, 'name', name)