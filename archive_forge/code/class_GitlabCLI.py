import argparse
import operator
import sys
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union
import gitlab
import gitlab.base
import gitlab.v4.objects
from gitlab import cli
from gitlab.exceptions import GitlabCiLintError
class GitlabCLI:

    def __init__(self, gl: gitlab.Gitlab, gitlab_resource: str, resource_action: str, args: Dict[str, str]) -> None:
        self.cls: Type[gitlab.base.RESTObject] = cli.gitlab_resource_to_cls(gitlab_resource, namespace=gitlab.v4.objects)
        self.cls_name = self.cls.__name__
        self.gitlab_resource = gitlab_resource.replace('-', '_')
        self.resource_action = resource_action.lower()
        self.gl = gl
        self.args = args
        self.parent_args: Dict[str, Any] = {}
        self.mgr_cls: Union[Type[gitlab.mixins.CreateMixin], Type[gitlab.mixins.DeleteMixin], Type[gitlab.mixins.GetMixin], Type[gitlab.mixins.GetWithoutIdMixin], Type[gitlab.mixins.ListMixin], Type[gitlab.mixins.UpdateMixin]] = getattr(gitlab.v4.objects, f'{self.cls.__name__}Manager')
        if TYPE_CHECKING:
            assert self.mgr_cls._path is not None
        self._process_from_parent_attrs()
        self.mgr_cls._path = self.mgr_cls._path.format(**self.parent_args)
        self.mgr = self.mgr_cls(gl)
        self.mgr._from_parent_attrs = self.parent_args
        if self.mgr_cls._types:
            for attr_name, type_cls in self.mgr_cls._types.items():
                if attr_name in self.args.keys():
                    obj = type_cls()
                    obj.set_from_cli(self.args[attr_name])
                    self.args[attr_name] = obj.get()

    def _process_from_parent_attrs(self) -> None:
        """Items in the path need to be url-encoded. There is a 1:1 mapping from
        mgr_cls._from_parent_attrs <--> mgr_cls._path. Those values must be url-encoded
        as they may contain a slash '/'."""
        for key in self.mgr_cls._from_parent_attrs:
            if key not in self.args:
                continue
            self.parent_args[key] = gitlab.utils.EncodedId(self.args[key])
            del self.args[key]

    def run(self) -> Any:
        method = f'do_{self.gitlab_resource}_{self.resource_action}'
        if hasattr(self, method):
            return getattr(self, method)()
        method = f'do_{self.resource_action}'
        if hasattr(self, method):
            return getattr(self, method)()
        return self.do_custom()

    def do_custom(self) -> Any:
        class_instance: Union[gitlab.base.RESTManager, gitlab.base.RESTObject]
        in_obj = cli.custom_actions[self.cls_name][self.resource_action][2]
        if in_obj:
            data = {}
            if self.mgr._from_parent_attrs:
                for k in self.mgr._from_parent_attrs:
                    data[k] = self.parent_args[k]
            if not issubclass(self.cls, gitlab.mixins.GetWithoutIdMixin):
                if TYPE_CHECKING:
                    assert isinstance(self.cls._id_attr, str)
                data[self.cls._id_attr] = self.args.pop(self.cls._id_attr)
            class_instance = self.cls(self.mgr, data)
        else:
            class_instance = self.mgr
        method_name = self.resource_action.replace('-', '_')
        return getattr(class_instance, method_name)(**self.args)

    def do_project_export_download(self) -> None:
        try:
            project = self.gl.projects.get(self.parent_args['project_id'], lazy=True)
            export_status = project.exports.get()
            if TYPE_CHECKING:
                assert export_status is not None
            data = export_status.download()
            if TYPE_CHECKING:
                assert data is not None
                assert isinstance(data, bytes)
            sys.stdout.buffer.write(data)
        except Exception as e:
            cli.die('Impossible to download the export', e)

    def do_validate(self) -> None:
        if TYPE_CHECKING:
            assert isinstance(self.mgr, gitlab.v4.objects.CiLintManager)
        try:
            self.mgr.validate(self.args)
        except GitlabCiLintError as e:
            cli.die('CI YAML Lint failed', e)
        except Exception as e:
            cli.die('Cannot validate CI YAML', e)

    def do_create(self) -> gitlab.base.RESTObject:
        if TYPE_CHECKING:
            assert isinstance(self.mgr, gitlab.mixins.CreateMixin)
        try:
            result = self.mgr.create(self.args)
        except Exception as e:
            cli.die('Impossible to create object', e)
        return result

    def do_list(self) -> Union[gitlab.base.RESTObjectList, List[gitlab.base.RESTObject]]:
        if TYPE_CHECKING:
            assert isinstance(self.mgr, gitlab.mixins.ListMixin)
        try:
            result = self.mgr.list(**self.args)
        except Exception as e:
            cli.die('Impossible to list objects', e)
        return result

    def do_get(self) -> Optional[gitlab.base.RESTObject]:
        if isinstance(self.mgr, gitlab.mixins.GetWithoutIdMixin):
            try:
                result = self.mgr.get(id=None, **self.args)
            except Exception as e:
                cli.die('Impossible to get object', e)
            return result
        if TYPE_CHECKING:
            assert isinstance(self.mgr, gitlab.mixins.GetMixin)
            assert isinstance(self.cls._id_attr, str)
        id = self.args.pop(self.cls._id_attr)
        try:
            result = self.mgr.get(id, lazy=False, **self.args)
        except Exception as e:
            cli.die('Impossible to get object', e)
        return result

    def do_delete(self) -> None:
        if TYPE_CHECKING:
            assert isinstance(self.mgr, gitlab.mixins.DeleteMixin)
            assert isinstance(self.cls._id_attr, str)
        id = self.args.pop(self.cls._id_attr)
        try:
            self.mgr.delete(id, **self.args)
        except Exception as e:
            cli.die('Impossible to destroy object', e)

    def do_update(self) -> Dict[str, Any]:
        if TYPE_CHECKING:
            assert isinstance(self.mgr, gitlab.mixins.UpdateMixin)
        if issubclass(self.mgr_cls, gitlab.mixins.GetWithoutIdMixin):
            id = None
        else:
            if TYPE_CHECKING:
                assert isinstance(self.cls._id_attr, str)
            id = self.args.pop(self.cls._id_attr)
        try:
            result = self.mgr.update(id, self.args)
        except Exception as e:
            cli.die('Impossible to update object', e)
        return result