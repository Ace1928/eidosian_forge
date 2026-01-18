import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
class JSONPrinter:

    @staticmethod
    def display(d: Union[str, Dict[str, Any]], **_kwargs: Any) -> None:
        import json
        print(json.dumps(d))

    @staticmethod
    def display_list(data: List[Union[str, Dict[str, Any], gitlab.base.RESTObject]], fields: List[str], **_kwargs: Any) -> None:
        import json
        print(json.dumps([get_dict(obj, fields) for obj in data]))