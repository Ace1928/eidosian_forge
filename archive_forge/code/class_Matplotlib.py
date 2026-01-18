from ase.io.utils import PlottingVariables, make_patch_list
class Matplotlib(PlottingVariables):

    def __init__(self, atoms, ax, rotation='', radii=None, colors=None, scale=1, offset=(0, 0), **parameters):
        PlottingVariables.__init__(self, atoms, rotation=rotation, radii=radii, colors=colors, scale=scale, extra_offset=offset, **parameters)
        self.ax = ax
        self.figure = ax.figure
        self.ax.set_aspect('equal')

    def write(self):
        self.write_body()
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def write_body(self):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            self.ax.add_patch(patch)