from kivy.properties import ObjectProperty
from kivy.event import EventDispatcher
from collections import defaultdict
class RecycleDataViewBehavior(object):
    """A optional base class for data views (:attr:`RecycleView`.viewclass).
    If a view inherits from this class, the class's functions will be called
    when the view needs to be updated due to a data change or layout update.
    """

    def refresh_view_attrs(self, rv, index, data):
        """Called by the :class:`RecycleAdapter` when the view is initially
        populated with the values from the `data` dictionary for this item.

        Any pos or size info should be removed because they are set
        subsequently with :attr:`refresh_view_layout`.

        :Parameters:

            `rv`: :class:`RecycleView` instance
                The :class:`RecycleView` that caused the update.
            `data`: dict
                The data dict used to populate this view.
        """
        sizing_attrs = RecycleDataAdapter._sizing_attrs
        for key, value in data.items():
            if key not in sizing_attrs:
                setattr(self, key, value)

    def refresh_view_layout(self, rv, index, layout, viewport):
        """Called when the view's size is updated by the layout manager,
        :class:`RecycleLayoutManagerBehavior`.

        :Parameters:

            `rv`: :class:`RecycleView` instance
                The :class:`RecycleView` that caused the update.
            `viewport`: 4-tuple
                The coordinates of the bottom left and width height in layout
                manager coordinates. This may be larger than this view item.

        :raises:
            `LayoutChangeException`: If the sizing or data changed during a
            call to this method, raising a `LayoutChangeException` exception
            will force a refresh. Useful when data changed and we don't want
            to layout further since it'll be overwritten again soon.
        """
        w, h = layout.pop('size')
        if w is None:
            if h is not None:
                self.height = h
        elif h is None:
            self.width = w
        else:
            self.size = (w, h)
        for name, value in layout.items():
            setattr(self, name, value)

    def apply_selection(self, rv, index, is_selected):
        pass