import tkinter
import math
import sys
from tkinter import ttk
from .gui_utilities import UniformDictController, FpsLabelUpdater
from .raytracing_view import *
from .hyperboloid_utilities import unit_3_vector_and_distance_to_O13_hyperbolic_translation
from .zoom_slider import Slider, ZoomSlider
class FiniteViewer(ttk.Frame):

    def __init__(self, container, manifold, fillings_changed_callback=None, weights=None, cohomology_basis=None, cohomology_class=None):
        ttk.Frame.__init__(self, container)
        self.bindtags(self.bindtags() + ('finite',))
        self.fillings_changed_callback = fillings_changed_callback
        self.has_weights = weights and any(weights)
        main_frame = self.create_frame_with_main_widget(self, manifold, weights, cohomology_basis, cohomology_class)
        self.filling_dict = {'fillings': self._fillings_from_manifold()}
        row = 0
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=row, column=0, sticky=tkinter.NSEW, padx=0, pady=0, ipady=0)
        self.notebook.add(self.create_cusp_areas_frame(self), text='Cusp areas')
        self.notebook.add(self.create_fillings_frame(self), text='Fillings')
        self.notebook.add(self.create_skeleton_frame(self), text='Skeleton')
        self.notebook.add(self.create_quality_frame(self), text='Quality')
        self.notebook.add(self.create_light_frame(self), text='Light')
        self.notebook.add(self.create_navigation_frame(self), text='Navigation')
        self.notebook.bind('<<NotebookTabChanged>>', self.focus_viewer)
        row += 1
        main_frame.grid(row=row, column=0, sticky=tkinter.NSEW)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(row, weight=1)
        row += 1
        status_frame = self.create_status_frame(self)
        status_frame.grid(row=row, column=0, sticky=tkinter.NSEW)
        UniformDictController(self.widget.ui_parameter_dict, 'fov', update_function=self.widget.redraw_if_initialized, scale=self.fov_scale, label=self.fov_label, format_string='%.1f')
        self.widget.report_time_callback = FpsLabelUpdater(self.fps_label)
        self.menubar = None
        self.build_menus()
        if isinstance(container, tkinter.Toplevel) and self.menubar:
            container.config(menu=self.menubar)
        self.focus_viewer()

    def focus_viewer(self, event=None):
        self.widget.focus_set()

    def create_cusp_areas_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        row = 0
        cusp_area_maximum = 1.05 * _maximal_cusp_area(self.widget.manifold)
        for i in range(self.widget.manifold.num_cusps()):
            UniformDictController.create_horizontal_scale(frame, uniform_dict=self.widget.ui_parameter_dict, key='cuspAreas', title='Cusp %d' % i, left_end=0.0, right_end=cusp_area_maximum, row=row, update_function=self.widget.recompute_raytracing_data_and_redraw, index=i)
            row += 1
        frame.rowconfigure(row, weight=1)
        UniformDictController.create_checkbox(frame, self.widget.ui_parameter_dict, 'perspectiveType', update_function=self.checkbox_update, text='Ideal view', row=row, column=1)
        return frame

    def checkbox_update(self):
        self.widget.redraw_if_initialized()
        self.focus_viewer()

    def create_fillings_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        frame.columnconfigure(3, weight=1)
        frame.columnconfigure(4, weight=0)
        row = 0
        self.filling_controllers = []
        for i in range(self.widget.manifold.num_cusps()):
            self.filling_controllers.append(UniformDictController.create_horizontal_scale(frame, self.filling_dict, key='fillings', column=0, index=i, component_index=0, title='Cusp %d' % i, row=row, left_end=-15, right_end=15, update_function=self.push_fillings_to_manifold, scale_class=ZoomSlider))
            self.filling_controllers.append(UniformDictController.create_horizontal_scale(frame, self.filling_dict, key='fillings', column=3, index=i, component_index=1, title=None, row=row, left_end=-15, right_end=15, update_function=self.push_fillings_to_manifold, scale_class=ZoomSlider))
            row += 1
        frame.rowconfigure(row, weight=1)
        subframe = ttk.Frame(frame)
        subframe.grid(row=row, column=0, columnspan=5)
        subframe.columnconfigure(0, weight=1)
        subframe.columnconfigure(1, weight=0)
        subframe.columnconfigure(2, weight=0)
        subframe.columnconfigure(3, weight=1)
        recompute_button = ttk.Button(subframe, text='Recompute hyp. structure', command=self.recompute_hyperbolic_structure)
        recompute_button.grid(row=0, column=1)
        snap_button = ttk.Button(subframe, text='Round to integers', command=self.round_fillings)
        snap_button.grid(row=0, column=2)
        return frame

    def create_skeleton_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        row = 0
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='edgeThickness', title='Face boundary thickness', row=row, left_end=0.0, right_end=0.35, update_function=self.widget.redraw_if_initialized, format_string='%.3f')
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_parameter_dict, key='vertexRadius', title='Vertex Radius', row=row, left_end=0.0, right_end=1.35, update_function=self.widget.redraw_if_initialized, format_string='%.3f')
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_parameter_dict, key='edgeTubeRadius', title='Edge thickness', row=row, left_end=0.0, right_end=0.5, update_function=self.widget.redraw_if_initialized)
        return frame

    def create_quality_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        row = 0
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='maxSteps', title='Max Steps', row=row, left_end=1, right_end=100, update_function=self.widget.redraw_if_initialized)
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='maxDist', title='Max Distance', row=row, left_end=1.0, right_end=28.0, update_function=self.widget.redraw_if_initialized)
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='subpixelCount', title='Subpixel count', row=row, left_end=1.0, right_end=4.25, update_function=self.widget.redraw_if_initialized)
        return frame

    def create_light_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        row = 0
        if self.has_weights:
            UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='contrast', title='Contrast', row=row, left_end=0.0, right_end=0.25, update_function=self.widget.redraw_if_initialized, format_string='%.3f')
            row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='lightBias', title='Light bias', row=row, left_end=0.3, right_end=4.0, update_function=self.widget.redraw_if_initialized)
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='lightFalloff', title='Light falloff', row=row, left_end=0.1, right_end=2.0, update_function=self.widget.redraw_if_initialized)
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='brightness', title='Brightness', row=row, left_end=0.3, right_end=3.0, update_function=self.widget.redraw_if_initialized)
        return frame

    def create_navigation_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        frame.columnconfigure(3, weight=0)
        row = 0
        UniformDictController.create_horizontal_scale(frame, self.widget.navigation_dict, key='translationVelocity', title='Translation Speed', row=row, left_end=0.1, right_end=1.0)
        label = ttk.Label(frame, text='Keys: wasdec')
        label.grid(row=row, column=3, sticky=tkinter.NSEW)
        row += 1
        UniformDictController.create_horizontal_scale(frame, self.widget.navigation_dict, key='rotationVelocity', title='Rotation Speed', row=row, left_end=0.1, right_end=1.0)
        label = ttk.Label(frame, text=u'Keys: ←↑→↓xz')
        label.grid(row=row, column=3, sticky=tkinter.NSEW)
        row += 1
        label = ttk.Label(frame, text=_mouse_gestures_text())
        label.grid(row=row, column=0, columnspan=4)
        return frame

    def create_frame_with_main_widget(self, parent, manifold, weights, cohomology_basis, cohomology_class):
        frame = ttk.Frame(parent)
        column = 0
        self.widget = RaytracingView('finite', manifold, weights=weights, cohomology_basis=cohomology_basis, cohomology_class=cohomology_class, geodesics=[], container=frame, width=600, height=500, double=1, depth=1)
        self.widget.grid(row=0, column=column, sticky=tkinter.NSEW)
        self.widget.make_current()
        frame.columnconfigure(column, weight=1)
        frame.rowconfigure(0, weight=1)
        column += 1
        self.fov_scale = Slider(frame, left_end=20, right_end=120, orient=tkinter.VERTICAL)
        self.fov_scale.grid(row=0, column=column, sticky=tkinter.NSEW)
        return frame

    def create_status_frame(self, parent):
        frame = ttk.Frame(parent)
        column = 0
        label = ttk.Label(frame, text='FOV:')
        label.grid(row=0, column=column)
        column += 1
        self.fov_label = ttk.Label(frame)
        self.fov_label.grid(row=0, column=column)
        column += 1
        self.vol_label = ttk.Label(frame)
        self.vol_label.grid(row=0, column=column)
        column += 1
        self.fps_label = ttk.Label(frame)
        self.fps_label.grid(row=0, column=column)
        return frame

    def update_volume_label(self):
        try:
            vol_text = '%.3f' % self.widget.manifold.volume()
        except ValueError:
            vol_text = '-'
        sol_type = self.widget.manifold.solution_type(enum=True)
        sol_text = _solution_type_text[sol_type]
        try:
            self.vol_label.configure(text='Vol: %s (%s)' % (vol_text, sol_text))
        except AttributeError:
            pass

    def update_filling_sliders(self):
        for filling_controller in self.filling_controllers:
            filling_controller.update()

    def _fillings_from_manifold(self):
        return ['vec2[]', [[d['filling'][0], d['filling'][1]] for d in self.widget.manifold.cusp_info()]]

    def pull_fillings_from_manifold(self):
        self.filling_dict['fillings'] = self._fillings_from_manifold()
        self.update_filling_sliders()
        self.widget.recompute_raytracing_data_and_redraw()

    def push_fillings_to_manifold(self):
        self.widget.manifold.dehn_fill(self.filling_dict['fillings'][1])
        self.widget.recompute_raytracing_data_and_redraw()
        if self.fillings_changed_callback:
            self.fillings_changed_callback()

    def recompute_hyperbolic_structure(self):
        self.widget.manifold.init_hyperbolic_structure(force_recompute=True)
        self.widget.recompute_raytracing_data_and_redraw()
        self.update_volume_label()
        if self.fillings_changed_callback:
            self.fillings_changed_callback()

    def round_fillings(self):
        for f in self.filling_dict['fillings'][1]:
            for i in [0, 1]:
                f[i] = float(round(f[i]))
        self.update_filling_sliders()
        self.push_fillings_to_manifold()

    def build_menus(self):
        pass