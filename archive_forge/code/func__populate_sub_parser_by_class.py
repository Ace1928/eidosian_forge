import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
def _populate_sub_parser_by_class(cls: Type[gitlab.base.RESTObject], sub_parser: _SubparserType) -> None:
    mgr_cls_name = f'{cls.__name__}Manager'
    mgr_cls = getattr(gitlab.v4.objects, mgr_cls_name)
    action_parsers: Dict[str, argparse.ArgumentParser] = {}
    for action_name in ['list', 'get', 'create', 'update', 'delete']:
        if not hasattr(mgr_cls, action_name):
            continue
        sub_parser_action = sub_parser.add_parser(action_name, conflict_handler='resolve')
        action_parsers[action_name] = sub_parser_action
        sub_parser_action.add_argument('--sudo', required=False)
        if mgr_cls._from_parent_attrs:
            for x in mgr_cls._from_parent_attrs:
                sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
        if action_name == 'list':
            for x in mgr_cls._list_filters:
                sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=False)
            sub_parser_action.add_argument('--page', required=False, type=int)
            sub_parser_action.add_argument('--per-page', required=False, type=int)
            sub_parser_action.add_argument('--get-all', required=False, action='store_true', help='Return all items from the server, without pagination.')
        if action_name == 'delete':
            if cls._id_attr is not None:
                id_attr = cls._id_attr.replace('_', '-')
                sub_parser_action.add_argument(f'--{id_attr}', required=True)
        if action_name == 'get':
            if not issubclass(cls, gitlab.mixins.GetWithoutIdMixin):
                if cls._id_attr is not None:
                    id_attr = cls._id_attr.replace('_', '-')
                    sub_parser_action.add_argument(f'--{id_attr}', required=True)
            for x in mgr_cls._optional_get_attrs:
                sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=False)
        if action_name == 'create':
            for x in mgr_cls._create_attrs.required:
                sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
            for x in mgr_cls._create_attrs.optional:
                sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=False)
        if action_name == 'update':
            if cls._id_attr is not None:
                id_attr = cls._id_attr.replace('_', '-')
                sub_parser_action.add_argument(f'--{id_attr}', required=True)
            for x in mgr_cls._update_attrs.required:
                if x != cls._id_attr:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
            for x in mgr_cls._update_attrs.optional:
                if x != cls._id_attr:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=False)
    if cls.__name__ in cli.custom_actions:
        name = cls.__name__
        for action_name in cli.custom_actions[name]:
            action_parser = action_parsers.get(action_name)
            if action_parser is None:
                sub_parser_action = sub_parser.add_parser(action_name)
            else:
                sub_parser_action = action_parser
            if mgr_cls._from_parent_attrs:
                for x in mgr_cls._from_parent_attrs:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
                sub_parser_action.add_argument('--sudo', required=False)
            if not issubclass(cls, gitlab.mixins.GetWithoutIdMixin):
                if cls._id_attr is not None:
                    id_attr = cls._id_attr.replace('_', '-')
                    sub_parser_action.add_argument(f'--{id_attr}', required=True)
            required, optional, dummy = cli.custom_actions[name][action_name]
            for x in required:
                if x != cls._id_attr:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
            for x in optional:
                if x != cls._id_attr:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=False)
    if mgr_cls.__name__ in cli.custom_actions:
        name = mgr_cls.__name__
        for action_name in cli.custom_actions[name]:
            action_parser = action_parsers.get(action_name)
            if action_parser is None:
                sub_parser_action = sub_parser.add_parser(action_name)
            else:
                sub_parser_action = action_parser
            if mgr_cls._from_parent_attrs:
                for x in mgr_cls._from_parent_attrs:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
                sub_parser_action.add_argument('--sudo', required=False)
            required, optional, dummy = cli.custom_actions[name][action_name]
            for x in required:
                if x != cls._id_attr:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=True)
            for x in optional:
                if x != cls._id_attr:
                    sub_parser_action.add_argument(f'--{x.replace('_', '-')}', required=False)