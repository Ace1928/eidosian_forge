from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
class Configurable(HasTraits):
    config = Instance(Config, (), {})
    parent = Instance('traitlets.config.configurable.Configurable', allow_none=True)

    def __init__(self, **kwargs: t.Any) -> None:
        """Create a configurable given a config config.

        Parameters
        ----------
        config : Config
            If this is empty, default values are used. If config is a
            :class:`Config` instance, it will be used to configure the
            instance.
        parent : Configurable instance, optional
            The parent Configurable instance of this object.

        Notes
        -----
        Subclasses of Configurable must call the :meth:`__init__` method of
        :class:`Configurable` *before* doing anything else and using
        :func:`super`::

            class MyConfigurable(Configurable):
                def __init__(self, config=None):
                    super(MyConfigurable, self).__init__(config=config)
                    # Then any other code you need to finish initialization.

        This ensures that instances will be configured properly.
        """
        parent = kwargs.pop('parent', None)
        if parent is not None:
            if kwargs.get('config', None) is None:
                kwargs['config'] = parent.config
            self.parent = parent
        config = kwargs.pop('config', None)
        super().__init__(**kwargs)
        config_override_names = set()

        def notice_config_override(change: Bunch) -> None:
            """Record traits set by both config and kwargs.

            They will need to be overridden again after loading config.
            """
            if change.name in kwargs:
                config_override_names.add(change.name)
        self.observe(notice_config_override)
        if config is not None:
            self.config = config
        else:
            self._load_config(self.config)
        self.unobserve(notice_config_override)
        for name in config_override_names:
            setattr(self, name, kwargs[name])

    @classmethod
    def section_names(cls) -> list[str]:
        """return section names as a list"""
        return [c.__name__ for c in reversed(cls.__mro__) if issubclass(c, Configurable) and issubclass(cls, c)]

    def _find_my_config(self, cfg: Config) -> t.Any:
        """extract my config from a global Config object

        will construct a Config object of only the config values that apply to me
        based on my mro(), as well as those of my parent(s) if they exist.

        If I am Bar and my parent is Foo, and their parent is Tim,
        this will return merge following config sections, in this order::

            [Bar, Foo.Bar, Tim.Foo.Bar]

        With the last item being the highest priority.
        """
        cfgs = [cfg]
        if self.parent:
            cfgs.append(self.parent._find_my_config(cfg))
        my_config = Config()
        for c in cfgs:
            for sname in self.section_names():
                if c._has_section(sname):
                    my_config.merge(c[sname])
        return my_config

    def _load_config(self, cfg: Config, section_names: list[str] | None=None, traits: dict[str, TraitType[t.Any, t.Any]] | None=None) -> None:
        """load traits from a Config object"""
        if traits is None:
            traits = self.traits(config=True)
        if section_names is None:
            section_names = self.section_names()
        my_config = self._find_my_config(cfg)
        with self.hold_trait_notifications():
            for name, config_value in my_config.items():
                if name in traits:
                    if isinstance(config_value, LazyConfigValue):
                        initial = getattr(self, name)
                        config_value = config_value.get_value(initial)
                    elif isinstance(config_value, DeferredConfig):
                        config_value = config_value.get_value(traits[name])
                    setattr(self, name, deepcopy(config_value))
                elif not _is_section_key(name) and (not isinstance(config_value, Config)):
                    from difflib import get_close_matches
                    if isinstance(self, LoggingConfigurable):
                        assert self.log is not None
                        warn = self.log.warning
                    else:

                        def warn(msg: t.Any) -> None:
                            return warnings.warn(msg, UserWarning, stacklevel=9)
                    matches = get_close_matches(name, traits)
                    msg = f'Config option `{name}` not recognized by `{self.__class__.__name__}`.'
                    if len(matches) == 1:
                        msg += f'  Did you mean `{matches[0]}`?'
                    elif len(matches) >= 1:
                        msg += '  Did you mean one of: `{matches}`?'.format(matches=', '.join(sorted(matches)))
                    warn(msg)

    @observe('config')
    @observe_compat
    def _config_changed(self, change: Bunch) -> None:
        """Update all the class traits having ``config=True`` in metadata.

        For any class trait with a ``config`` metadata attribute that is
        ``True``, we update the trait with the value of the corresponding
        config entry.
        """
        traits = self.traits(config=True)
        section_names = self.section_names()
        self._load_config(change.new, traits=traits, section_names=section_names)

    def update_config(self, config: Config) -> None:
        """Update config and load the new values"""
        self.config = deepcopy(self.config)
        self._load_config(config)
        self.config.merge(config)

    @classmethod
    def class_get_help(cls, inst: HasTraits | None=None) -> str:
        """Get the help string for this class in ReST format.

        If `inst` is given, its current trait values will be used in place of
        class defaults.
        """
        assert inst is None or isinstance(inst, cls)
        final_help = []
        base_classes = ', '.join((p.__name__ for p in cls.__bases__))
        final_help.append(f'{cls.__name__}({base_classes}) options')
        final_help.append(len(final_help[0]) * '-')
        for _, v in sorted(cls.class_traits(config=True).items()):
            help = cls.class_get_trait_help(v, inst)
            final_help.append(help)
        return '\n'.join(final_help)

    @classmethod
    def class_get_trait_help(cls, trait: TraitType[t.Any, t.Any], inst: HasTraits | None=None, helptext: str | None=None) -> str:
        """Get the helptext string for a single trait.

        :param inst:
            If given, its current trait values will be used in place of
            the class default.
        :param helptext:
            If not given, uses the `help` attribute of the current trait.
        """
        assert inst is None or isinstance(inst, cls)
        lines = []
        header = f'--{cls.__name__}.{trait.name}'
        if isinstance(trait, (Container, Dict)):
            multiplicity = trait.metadata.get('multiplicity', 'append')
            if isinstance(trait, Dict):
                sample_value = '<key-1>=<value-1>'
            else:
                sample_value = '<%s-item-1>' % trait.__class__.__name__.lower()
            if multiplicity == 'append':
                header = f'{header}={sample_value}...'
            else:
                header = f'{header} {sample_value}...'
        else:
            header = f'{header}=<{trait.__class__.__name__}>'
        lines.append(header)
        if helptext is None:
            helptext = trait.help
        if helptext != '':
            helptext = '\n'.join(wrap_paragraphs(helptext, 76))
            lines.append(indent(helptext))
        if 'Enum' in trait.__class__.__name__:
            lines.append(indent('Choices: %s' % trait.info()))
        if inst is not None:
            lines.append(indent(f'Current: {getattr(inst, trait.name or '')!r}'))
        else:
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + '...'
                lines.append(indent('Default: %s' % dvr))
        return '\n'.join(lines)

    @classmethod
    def class_print_help(cls, inst: HasTraits | None=None) -> None:
        """Get the help string for a single trait and print it."""
        print(cls.class_get_help(inst))

    @classmethod
    def _defining_class(cls, trait: TraitType[t.Any, t.Any], classes: t.Sequence[type[HasTraits]]) -> type[Configurable]:
        """Get the class that defines a trait

        For reducing redundant help output in config files.
        Returns the current class if:
        - the trait is defined on this class, or
        - the class where it is defined would not be in the config file

        Parameters
        ----------
        trait : Trait
            The trait to look for
        classes : list
            The list of other classes to consider for redundancy.
            Will return `cls` even if it is not defined on `cls`
            if the defining class is not in `classes`.
        """
        defining_cls = cls
        assert trait.name is not None
        for parent in cls.mro():
            if issubclass(parent, Configurable) and parent in classes and (parent.class_own_traits(config=True).get(trait.name, None) is trait):
                defining_cls = parent
        return defining_cls

    @classmethod
    def class_config_section(cls, classes: t.Sequence[type[HasTraits]] | None=None) -> str:
        """Get the config section for this class.

        Parameters
        ----------
        classes : list, optional
            The list of other classes in the config file.
            Used to reduce redundant information.
        """

        def c(s: str) -> str:
            """return a commented, wrapped block."""
            s = '\n\n'.join(wrap_paragraphs(s, 78))
            return '## ' + s.replace('\n', '\n#  ')
        breaker = '#' + '-' * 78
        parent_classes = ', '.join((p.__name__ for p in cls.__bases__ if issubclass(p, Configurable)))
        s = f'# {cls.__name__}({parent_classes}) configuration'
        lines = [breaker, s, breaker]
        desc = cls.class_traits().get('description')
        if desc:
            desc = desc.default_value
        if not desc:
            desc = getattr(cls, '__doc__', '')
        if desc:
            lines.append(c(desc))
            lines.append('')
        for name, trait in sorted(cls.class_traits(config=True).items()):
            default_repr = trait.default_value_repr()
            if classes:
                defining_class = cls._defining_class(trait, classes)
            else:
                defining_class = cls
            if defining_class is cls:
                if trait.help:
                    lines.append(c(trait.help))
                if 'Enum' in type(trait).__name__:
                    lines.append('#  Choices: %s' % trait.info())
                lines.append('#  Default: %s' % default_repr)
            else:
                if trait.help:
                    lines.append(c(trait.help.split('\n', 1)[0]))
                lines.append(f'#  See also: {defining_class.__name__}.{name}')
            lines.append(f'# c.{cls.__name__}.{name} = {default_repr}')
            lines.append('')
        return '\n'.join(lines)

    @classmethod
    def class_config_rst_doc(cls) -> str:
        """Generate rST documentation for this class' config options.

        Excludes traits defined on parent classes.
        """
        lines = []
        classname = cls.__name__
        for _, trait in sorted(cls.class_traits(config=True).items()):
            ttype = trait.__class__.__name__
            if not trait.name:
                continue
            termline = classname + '.' + trait.name
            if 'Enum' in ttype:
                termline += ' : ' + trait.info_rst()
            else:
                termline += ' : ' + ttype
            lines.append(termline)
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + '...'
                dvr = dvr.replace('\\n', '\\\\n')
                lines.append(indent('Default: ``%s``' % dvr))
                lines.append('')
            help = trait.help or 'No description'
            lines.append(indent(dedent(help)))
            lines.append('')
        return '\n'.join(lines)