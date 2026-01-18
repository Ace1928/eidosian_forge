import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
class YAMLPrinter:

    @staticmethod
    def display(d: Union[str, Dict[str, Any]], **_kwargs: Any) -> None:
        try:
            import yaml
            print(yaml.safe_dump(d, default_flow_style=False))
        except ImportError:
            sys.exit('PyYaml is not installed.\nInstall it with `pip install PyYaml` to use the yaml output feature')

    @staticmethod
    def display_list(data: List[Union[str, Dict[str, Any], gitlab.base.RESTObject]], fields: List[str], **_kwargs: Any) -> None:
        try:
            import yaml
            print(yaml.safe_dump([get_dict(obj, fields) for obj in data], default_flow_style=False))
        except ImportError:
            sys.exit('PyYaml is not installed.\nInstall it with `pip install PyYaml` to use the yaml output feature')