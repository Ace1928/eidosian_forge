import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
class OptionTree(AttrTree):
    """
    A subclass of AttrTree that is used to define the inheritance
    relationships between a collection of Options objects. Each node
    of the tree supports a group of Options objects and the leaf nodes
    inherit their keyword values from parent nodes up to the root.

    Supports the ability to search the tree for the closest valid path
    using the find method, or compute the appropriate Options value
    given an object and a mode. For a given node of the tree, the
    options method computes a Options object containing the result of
    inheritance for a given group up to the root of the tree.

    When constructing an OptionTree, you can specify the option groups
    as a list (i.e. empty initial option groups at the root) or as a
    dictionary (e.g. groups={'style':Option()}). You can also
    initialize the OptionTree with the options argument together with
    the **kwargs - see StoreOptions.merge_options for more information
    on the options specification syntax.

    You can use the string specifier '.' to refer to the root node in
    the options specification. This acts as an alternative was of
    specifying the options groups of the current node. Note that this
    approach method may only be used with the group lists format.
    """

    def __init__(self, items=None, identifier=None, parent=None, groups=None, options=None, backend=None, **kwargs):
        if groups is None:
            raise ValueError('Please supply groups list or dictionary')
        _groups = {g: Options() for g in groups} if isinstance(groups, list) else groups
        self.__dict__['backend'] = backend
        self.__dict__['groups'] = _groups
        self.__dict__['_instantiated'] = False
        AttrTree.__init__(self, items, identifier, parent)
        self.__dict__['_instantiated'] = True
        options = StoreOptions.merge_options(_groups.keys(), options, **kwargs)
        root_groups = options.pop('.', None)
        if root_groups and isinstance(groups, list):
            self.__dict__['groups'] = {g: Options(**root_groups.get(g, {})) for g in _groups.keys()}
        elif root_groups:
            raise Exception("Group specification as a dictionary only supported if the root node '.' syntax not used in the options.")
        if options:
            StoreOptions.apply_customizations(options, self)

    def _merge_options(self, identifier, group_name, options):
        """
        Computes a merged Options object for the given group
        name from the existing Options on the node and the
        new Options which are passed in.
        """
        if group_name not in self.groups:
            raise KeyError(f'Group {group_name} not defined on SettingTree.')
        if identifier in self.children:
            current_node = self[identifier]
            group_options = current_node.groups[group_name]
        else:
            group_options = Options(group_name, allowed_keywords=self.groups[group_name].allowed_keywords)
        override_kwargs = dict(options.kwargs)
        old_allowed = group_options.allowed_keywords
        override_kwargs['allowed_keywords'] = options.allowed_keywords + old_allowed
        try:
            if options.merge_keywords:
                return group_options(**override_kwargs)
            else:
                return Options(group_name, **override_kwargs)
        except OptionError as e:
            raise OptionError(e.invalid_keyword, e.allowed_keywords, group_name=group_name, path=self.path) from e

    def __getitem__(self, item):
        if item in self.groups:
            return self.groups[item]
        return super().__getitem__(item)

    def __getattr__(self, identifier):
        """
        Allows creating sub OptionTree instances using attribute
        access, inheriting the group options.
        """
        try:
            return super(AttrTree, self).__getattr__(identifier)
        except AttributeError:
            pass
        if identifier.startswith('_'):
            raise AttributeError(str(identifier))
        elif self.fixed == True:
            raise AttributeError(self._fixed_error % identifier)
        valid_id = sanitize_identifier(identifier, escape=False)
        if valid_id in self.children:
            return self.__dict__[valid_id]
        self.__setattr__(identifier, {k: Options(k, allowed_keywords=v.allowed_keywords) for k, v in self.groups.items()})
        return self[identifier]

    def __setattr__(self, identifier, val):
        Store._lookup_cache[self.backend] = {}
        identifier = sanitize_identifier(identifier, escape=False)
        new_groups = {}
        if isinstance(val, dict):
            group_items = val
        elif isinstance(val, Options) and val.key is None:
            raise AttributeError('Options object needs to have a group name specified.')
        elif isinstance(val, Options) and val.key[0].isupper():
            groups = ', '.join((repr(el) for el in Options._option_groups))
            raise AttributeError(f'OptionTree only accepts Options using keys that are one of {groups}.')
        elif isinstance(val, Options):
            group_items = {val.key: val}
        elif isinstance(val, OptionTree):
            group_items = val.groups
        current_node = self[identifier] if identifier in self.children else self
        for group_name in current_node.groups:
            options = group_items.get(group_name, False)
            if options:
                new_groups[group_name] = self._merge_options(identifier, group_name, options)
            else:
                new_groups[group_name] = current_node.groups[group_name]
        if new_groups:
            data = self[identifier].items() if identifier in self.children else None
            new_node = OptionTree(data, identifier=identifier, parent=self, groups=new_groups, backend=self.backend)
        else:
            raise ValueError('OptionTree only accepts a dictionary of Options.')
        super().__setattr__(identifier, new_node)
        if isinstance(val, OptionTree):
            for subtree in val:
                self[identifier].__setattr__(subtree.identifier, subtree)

    def find(self, path, mode='node'):
        """
        Find the closest node or path to an the arbitrary path that is
        supplied down the tree from the given node. The mode argument
        may be either 'node' or 'path' which determines the return
        type.
        """
        path = path.split('.') if isinstance(path, str) else list(path)
        item = self
        for child in path:
            escaped_child = sanitize_identifier(child, escape=False)
            matching_children = (c for c in item.children if child.endswith(c) or escaped_child.endswith(c))
            matching_children = sorted(matching_children, key=lambda x: -len(x))
            if matching_children:
                item = item[matching_children[0]]
            else:
                continue
        return item if mode == 'node' else item.path

    def closest(self, obj, group, defaults=True, backend=None):
        """
        This method is designed to be called from the root of the
        tree. Given any LabelledData object, this method will return
        the most appropriate Options object, including inheritance.

        In addition, closest supports custom options by checking the
        object
        """
        opts_spec = (obj.__class__.__name__, group_sanitizer(obj.group), label_sanitizer(obj.label))
        backend = backend or Store.current_backend
        cache = Store._lookup_cache.get(backend, {})
        cache_key = opts_spec + (group, defaults, id(self.root))
        if cache_key in cache:
            return cache[cache_key]
        target = '.'.join((c for c in opts_spec if c))
        options = self.find(opts_spec).options(group, target=target, defaults=defaults, backend=backend)
        cache[cache_key] = options
        return options

    def options(self, group, target=None, defaults=True, backend=None):
        """
        Using inheritance up to the root, get the complete Options
        object for the given node and the specified group.
        """
        if target is None:
            target = self.path
        if self.groups.get(group, None) is None:
            return None
        options = Store.options(backend=backend)
        if self.parent is None and target and (self is not options) and defaults:
            root_name = self.__class__.__name__
            replacement = root_name + ('' if len(target) == len(root_name) else '.')
            option_key = target.replace(replacement, '')
            match = options.find(option_key)
            if match is not options:
                return match.options(group)
            else:
                return EMPTY_OPTIONS
        elif self.parent is None:
            return self.groups[group]
        parent_opts = self.parent.options(group, target, defaults, backend=backend)
        return Options(**dict(parent_opts.kwargs, **self.groups[group].kwargs))

    def __repr__(self):
        """
        Evalable representation of the OptionTree.
        """
        groups = self.__dict__['groups']
        tab, gsep = ('   ', ',\n\n')
        esep, gspecs = (',\n' + tab * 2, [])
        for group in groups.keys():
            especs, accumulator = ([], [])
            if groups[group].kwargs != {}:
                accumulator.append(('.', groups[group].kwargs))
            for t, v in sorted(self.items()):
                kwargs = v.groups[group].kwargs
                accumulator.append(('.'.join(t), kwargs))
            for t, kws in accumulator:
                if group == 'norm' and all((kws.get(k, False) is False for k in ['axiswise', 'framewise'])):
                    continue
                elif kws:
                    especs.append((t, kws))
            if especs:
                format_kws = [(t, f'dict({', '.join((f'{k}={v}' for k, v in sorted(kws.items())))})') for t, kws in especs]
                ljust = max((len(t) for t, _ in format_kws))
                sep = tab * 2 if len(format_kws) > 1 else ''
                entries = sep + esep.join([f'{sep}{t.ljust(ljust)} : {v}' for t, v in format_kws])
                gspecs.append(('%s%s={\n%s}' if len(format_kws) > 1 else '%s%s={%s}') % (tab, group, entries))
        return f'OptionTree(groups={groups.keys()},\n{gsep.join(gspecs)}\n)'