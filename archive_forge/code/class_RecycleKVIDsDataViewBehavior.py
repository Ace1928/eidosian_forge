from kivy.properties import ObjectProperty
from kivy.event import EventDispatcher
from collections import defaultdict
class RecycleKVIDsDataViewBehavior(RecycleDataViewBehavior):
    """Similar to :class:`RecycleDataViewBehavior`, except that the data keys
    can signify properties of an object named with an id in the root KV rule.

    E.g. given a KV rule::

        <MyRule@RecycleKVIDsDataViewBehavior+BoxLayout>:
            Label:
                id: name
            Label:
                id: value

    Then setting the data list with
    ``rv.data = [{'name.text': 'Kivy user', 'value.text': '12'}]`` would
    automatically set the corresponding labels.

    So, if the key doesn't have a period, the named property of the root widget
    will be set to the corresponding value. If there is a period, the named
    property of the widget with the id listed before the period will be set to
    the corresponding value.

    .. versionadded:: 2.0.0
    """

    def refresh_view_attrs(self, rv, index, data):
        sizing_attrs = RecycleDataAdapter._sizing_attrs
        for key, value in data.items():
            if key not in sizing_attrs:
                name, *ids = key.split('.')
                if ids:
                    if len(ids) != 1:
                        raise ValueError(f'Data key "{key}" has more than one period')
                    setattr(self.ids[name], ids[0], value)
                else:
                    setattr(self, name, value)