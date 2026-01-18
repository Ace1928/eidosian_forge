from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
class HostAxesBase:

    def __init__(self, *args, **kwargs):
        self.parasites = []
        super().__init__(*args, **kwargs)

    def get_aux_axes(self, tr=None, viewlim_mode='equal', axes_class=None, **kwargs):
        """
        Add a parasite axes to this host.

        Despite this method's name, this should actually be thought of as an
        ``add_parasite_axes`` method.

        .. versionchanged:: 3.7
           Defaults to same base axes class as host axes.

        Parameters
        ----------
        tr : `~matplotlib.transforms.Transform` or None, default: None
            If a `.Transform`, the following relation will hold:
            ``parasite.transData = tr + host.transData``.
            If None, the parasite's and the host's ``transData`` are unrelated.
        viewlim_mode : {"equal", "transform", None}, default: "equal"
            How the parasite's view limits are set: directly equal to the
            parent axes ("equal"), equal after application of *tr*
            ("transform"), or independently (None).
        axes_class : subclass type of `~matplotlib.axes.Axes`, optional
            The `~.axes.Axes` subclass that is instantiated.  If None, the base
            class of the host axes is used.
        **kwargs
            Other parameters are forwarded to the parasite axes constructor.
        """
        if axes_class is None:
            axes_class = self._base_axes_class
        parasite_axes_class = parasite_axes_class_factory(axes_class)
        ax2 = parasite_axes_class(self, tr, viewlim_mode=viewlim_mode, **kwargs)
        self.parasites.append(ax2)
        ax2._remove_method = self.parasites.remove
        return ax2

    def draw(self, renderer):
        orig_children_len = len(self._children)
        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.set_position(pos, which='active')
            self.apply_aspect(pos)
        else:
            self.apply_aspect()
        rect = self.get_position()
        for ax in self.parasites:
            ax.apply_aspect(rect)
            self._children.extend(ax.get_children())
        super().draw(renderer)
        del self._children[orig_children_len:]

    def clear(self):
        super().clear()
        for ax in self.parasites:
            ax.clear()

    def pick(self, mouseevent):
        super().pick(mouseevent)
        for a in self.parasites:
            a.pick(mouseevent)

    def twinx(self, axes_class=None):
        """
        Create a twin of Axes with a shared x-axis but independent y-axis.

        The y-axis of self will have ticks on the left and the returned axes
        will have ticks on the right.
        """
        ax = self._add_twin_axes(axes_class, sharex=self)
        self.axis['right'].set_visible(False)
        ax.axis['right'].set_visible(True)
        ax.axis['left', 'top', 'bottom'].set_visible(False)
        return ax

    def twiny(self, axes_class=None):
        """
        Create a twin of Axes with a shared y-axis but independent x-axis.

        The x-axis of self will have ticks on the bottom and the returned axes
        will have ticks on the top.
        """
        ax = self._add_twin_axes(axes_class, sharey=self)
        self.axis['top'].set_visible(False)
        ax.axis['top'].set_visible(True)
        ax.axis['left', 'right', 'bottom'].set_visible(False)
        return ax

    def twin(self, aux_trans=None, axes_class=None):
        """
        Create a twin of Axes with no shared axis.

        While self will have ticks on the left and bottom axis, the returned
        axes will have ticks on the top and right axis.
        """
        if aux_trans is None:
            aux_trans = mtransforms.IdentityTransform()
        ax = self._add_twin_axes(axes_class, aux_transform=aux_trans, viewlim_mode='transform')
        self.axis['top', 'right'].set_visible(False)
        ax.axis['top', 'right'].set_visible(True)
        ax.axis['left', 'bottom'].set_visible(False)
        return ax

    def _add_twin_axes(self, axes_class, **kwargs):
        """
        Helper for `.twinx`/`.twiny`/`.twin`.

        *kwargs* are forwarded to the parasite axes constructor.
        """
        if axes_class is None:
            axes_class = self._base_axes_class
        ax = parasite_axes_class_factory(axes_class)(self, **kwargs)
        self.parasites.append(ax)
        ax._remove_method = self._remove_any_twin
        return ax

    def _remove_any_twin(self, ax):
        self.parasites.remove(ax)
        restore = ['top', 'right']
        if ax._sharex:
            restore.remove('top')
        if ax._sharey:
            restore.remove('right')
        self.axis[tuple(restore)].set_visible(True)
        self.axis[tuple(restore)].toggle(ticklabels=False, label=False)

    @_api.make_keyword_only('3.8', 'call_axes_locator')
    def get_tightbbox(self, renderer=None, call_axes_locator=True, bbox_extra_artists=None):
        bbs = [*[ax.get_tightbbox(renderer, call_axes_locator=call_axes_locator) for ax in self.parasites], super().get_tightbbox(renderer, call_axes_locator=call_axes_locator, bbox_extra_artists=bbox_extra_artists)]
        return Bbox.union([b for b in bbs if b.width != 0 or b.height != 0])