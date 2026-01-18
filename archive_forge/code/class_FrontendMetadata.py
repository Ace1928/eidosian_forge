from abc import ABCMeta
from abc import abstractmethod
class FrontendMetadata:
    """Metadata required to render a plugin on the frontend.

    Each argument to the constructor is publicly accessible under a
    field of the same name. See constructor docs for further details.
    """

    def __init__(self, *, disable_reload=None, element_name=None, es_module_path=None, remove_dom=None, tab_name=None, is_ng_component=None):
        """Creates a `FrontendMetadata` value.

        The argument list is sorted and may be extended in the future;
        therefore, callers must pass only named arguments to this
        constructor.

        Args:
          disable_reload: Whether to disable the reload button and
              auto-reload timer. A `bool`; defaults to `False`.
          element_name: For legacy plugins, name of the custom element
              defining the plugin frontend: e.g., `"tf-scalar-dashboard"`.
              A `str` or `None` (for iframed plugins). Mutually exclusive
              with `es_module_path`.
          es_module_path: ES module to use as an entry point to this plugin.
              A `str` that is a key in the result of `get_plugin_apps()`, or
              `None` for legacy plugins bundled with TensorBoard as part of
              `webfiles.zip`. Mutually exclusive with legacy `element_name`
          remove_dom: Whether to remove the plugin DOM when switching to a
              different plugin, to trigger the Polymer 'detached' event.
              A `bool`; defaults to `False`.
          tab_name: Name to show in the menu item for this dashboard within
              the navigation bar. May differ from the plugin name: for
              instance, the tab name should not use underscores to separate
              words. Should be a `str` or `None` (the default; indicates to
              use the plugin name as the tab name).
          is_ng_component: Set to `True` only for built-in Angular plugins.
              In this case, the `plugin_name` property of the Plugin, which is
              mapped to the `id` property in JavaScript's `UiPluginMetadata` type,
              is used to select the Angular component. A `True` value is mutually
              exclusive with `element_name` and `es_module_path`.
        """
        self._disable_reload = False if disable_reload is None else disable_reload
        self._element_name = element_name
        self._es_module_path = es_module_path
        self._remove_dom = False if remove_dom is None else remove_dom
        self._tab_name = tab_name
        self._is_ng_component = False if is_ng_component is None else is_ng_component

    @property
    def disable_reload(self):
        return self._disable_reload

    @property
    def element_name(self):
        return self._element_name

    @property
    def is_ng_component(self):
        return self._is_ng_component

    @property
    def es_module_path(self):
        return self._es_module_path

    @property
    def remove_dom(self):
        return self._remove_dom

    @property
    def tab_name(self):
        return self._tab_name

    def __eq__(self, other):
        if not isinstance(other, FrontendMetadata):
            return False
        if self._disable_reload != other._disable_reload:
            return False
        if self._disable_reload != other._disable_reload:
            return False
        if self._element_name != other._element_name:
            return False
        if self._es_module_path != other._es_module_path:
            return False
        if self._remove_dom != other._remove_dom:
            return False
        if self._tab_name != other._tab_name:
            return False
        return True

    def __hash__(self):
        return hash((self._disable_reload, self._element_name, self._es_module_path, self._remove_dom, self._tab_name, self._is_ng_component))

    def __repr__(self):
        return 'FrontendMetadata(%s)' % ', '.join(('disable_reload=%r' % self._disable_reload, 'element_name=%r' % self._element_name, 'es_module_path=%r' % self._es_module_path, 'remove_dom=%r' % self._remove_dom, 'tab_name=%r' % self._tab_name, 'is_ng_component=%r' % self._is_ng_component))