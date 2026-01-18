from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
class HyperboloidNavigation:
    """
    A mixin class for a Tk widget that binds some key and mouse events
    to navigate through the hyperboloid model of hyperbolic 3-space.

    This is a mixin class and some other class in the class hierarchy
    is expected to provide the following attributes and methods:

    - self.raytracing_data has to be an instance of, e.g.,
      IdealRaytracingData. This is needed to update data
      such as the view matrix
      using self.raytracing_data.update_view_state(...).
    - self.redraw_if_initialized() to redraw.
    - self.read_depth_value(x, y) to return the depth value at a pixel.
      It is used for orbiting about that point.
    - self.compute_translation_and_inverse_from_pick_point(size, xy, depth)
      returning the SO(1,3)-matrices for conjugating to orbit with a certain
      speed about the point with frag coord xy and depth given a viewport of
      size size.

    The mixin class will provide the attribute self.view_state (e.g.,
    pair of view matrix and tetrahedron we are in).
    """

    def __init__(self):
        self.mouse_pos_when_pressed = None
        self.view_state_when_pressed = None
        self.last_mouse_pos = None
        self.mouse_mode = None
        self.setup_keymapping()
        self.process_keys_and_redraw_scheduled = False
        self.view_state = self.raytracing_data.initial_view_state()
        self.cursor = _default_cursor
        self.configure(cursor=self.cursor)
        self.navigation_dict = {'translationVelocity': ['float', 0.4], 'rotationVelocity': ['float', 0.4]}
        self.bind('<KeyPress>', self.tkKeyPress)
        self.bind('<KeyRelease>', self.tkKeyRelease)
        self.bind_class('inside', '<KeyPress>', self.tkKeyPress)
        self.bind_class('inside', '<KeyRelease>', self.tkKeyRelease)
        self.bind('<Button-1>', self.tkButton1)
        self.bind('<Shift-Button-1>', self.tkShiftButton1)
        self.bind('<Alt-Button-1>', self.tkAltButton1)
        if sys.platform == 'darwin':
            self.bind('<Option-Button-1>', self.tkAltButton1)
        self.bind('<Command-Button-1>', self.tkAltButton1)
        self.bind('<B1-Motion>', self.tkButtonMotion1)
        self.bind('<ButtonRelease-1>', self.tkButtonRelease1)

    def reset_view_state(self):
        """
        Resets view state.
        """
        self.view_state = self.raytracing_data.initial_view_state()

    def fix_view_state(self):
        """
        Fixes view state. Implementation resides with self.raytracing_data,
        e.g., if the view matrix takes the camera outside of the current
        tetrahedron, it would change the view matrix and current tetrahedron
        to fix it.
        """
        self.view_state = self.raytracing_data.update_view_state(self.view_state)

    def schedule_process_key_events_and_redraw(self, time_ms):
        """
        Schedule call to process_key_events_and_redraw in given time
        (milliseconds) if not scheduled already.
        """
        if self.process_keys_and_redraw_scheduled:
            return
        self.process_keys_and_redraw_scheduled = True
        self.after(time_ms, self.process_key_events_and_redraw)

    def process_key_events_and_redraw(self):
        """
        Go through the recorded time stamps of the key press and release
        events and update the view accordingly.
        """
        self.process_keys_and_redraw_scheduled = False
        t = time.time()
        m = matrix.identity(self.raytracing_data.RF, 4)
        any_key = False
        for k, last_and_release in self.key_to_last_accounted_and_release_time.items():
            dT = None
            if last_and_release[0] is None:
                last_and_release[1] = None
            elif not last_and_release[1] is None and t - last_and_release[1] > _ignore_key_release_time_s:
                dT = last_and_release[1] - last_and_release[0]
                last_and_release[0] = None
                last_and_release[1] = None
            else:
                dT = t - last_and_release[0]
                last_and_release[0] = t
            if dT is not None:
                RF = m.base_ring()
                m = m * self.keymapping[k](RF(dT * self.navigation_dict['rotationVelocity'][1]), RF(dT * self.navigation_dict['translationVelocity'][1]))
                any_key = True
        if not any_key:
            return
        self.view_state = self.raytracing_data.update_view_state(self.view_state, m)
        self.redraw_if_initialized()
        self.schedule_process_key_events_and_redraw(_refresh_delay_ms)

    def tkKeyRelease(self, event):
        k = event.keysym.lower()
        t = time.time()
        last_and_release = self.key_to_last_accounted_and_release_time.get(k)
        if last_and_release:
            last_and_release[1] = t
        if k in _cursor_mappings:
            self.cursor = _default_cursor
        if not self.mouse_mode:
            self.configure(cursor=self.cursor)

    def tkKeyPress(self, event):
        if self.mouse_mode:
            return
        k = event.keysym.lower()
        t = time.time()
        cursor = _cursor_mappings.get(k)
        if cursor:
            self.configure(cursor=cursor)
        last_and_release = self.key_to_last_accounted_and_release_time.get(k)
        if last_and_release:
            if last_and_release[0] is None:
                last_and_release[0] = t
            last_and_release[1] = None
            self.schedule_process_key_events_and_redraw(1)
        if event.keysym == 'u':
            print('View SO(1,3)-matrix and current tetrahedron:', self.view_state)
        if event.keysym == 'v':
            self.view = (self.view + 1) % 3
            print('Color for rays that have not hit geometry:', _viewModes[self.view])
            self.redraw_if_initialized()
        if event.keysym == 'p':
            from snappy.CyOpenGL import get_gl_string
            self.make_current()
            for k in ['GL_VERSION', 'GL_SHADING_LANGUAGE_VERSION']:
                print('%s: %s' % (k, get_gl_string(k)))
        if event.keysym == 'm':
            width = 1000
            height = 1000
            f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            self.save_image(width, height, f)
            print('Image saved to: ', f.name)

    def tkButton1(self, event):
        for last, release in self.key_to_last_accounted_and_release_time.values():
            if last or release:
                return
        self.configure(cursor=_default_move_cursor)
        self.mouse_pos_when_pressed = (event.x, event.y)
        self.view_state_when_pressed = self.view_state
        self.mouse_mode = 'move'

    def tkShiftButton1(self, event):
        for last, release in self.key_to_last_accounted_and_release_time.values():
            if last or release:
                return
        self.mouse_pos_when_pressed = (event.x, event.y)
        self.view_state_when_pressed = self.view_state
        self.mouse_mode = 'rotate'

    def tkAltButton1(self, event):
        for last, release in self.key_to_last_accounted_and_release_time.values():
            if last or release:
                return
        self.make_current()
        depth, width, height = self.read_depth_value(event.x, event.y)
        self.orbit_translation, self.orbit_inv_translation, self.orbit_speed = self.compute_translation_and_inverse_from_pick_point((width, height), (event.x, height - event.y), depth)
        self.last_mouse_pos = (event.x, event.y)
        self.view_state_when_pressed = self.view_state
        self.orbit_rotation = matrix.identity(self.raytracing_data.RF, 4)
        self.mouse_mode = 'orbit'

    def tkButtonMotion1(self, event):
        if self.mouse_mode == 'orbit':
            delta_x = event.x - self.last_mouse_pos[0]
            delta_y = event.y - self.last_mouse_pos[1]
            RF = self.raytracing_data.RF
            angle_x = RF(delta_x * self.orbit_speed * 0.01)
            angle_y = RF(delta_y * self.orbit_speed * 0.01)
            m = O13_y_rotation(angle_x) * O13_x_rotation(angle_y)
            self.orbit_rotation = self.orbit_rotation * m
            self.view_state = self.raytracing_data.update_view_state(self.view_state_when_pressed, self.orbit_translation * self.orbit_rotation * self.orbit_inv_translation)
            self.last_mouse_pos = (event.x, event.y)
        elif self.mouse_mode == 'move':
            RF = self.raytracing_data.RF
            delta_x = RF(event.x - self.mouse_pos_when_pressed[0])
            delta_y = RF(event.y - self.mouse_pos_when_pressed[1])
            amt = (delta_x ** 2 + delta_y ** 2).sqrt()
            if amt == 0:
                self.view_state = self.view_state_when_pressed
            else:
                m = unit_3_vector_and_distance_to_O13_hyperbolic_translation([-delta_x / amt, delta_y / amt, RF(0)], amt * RF(0.01))
                self.view_state = self.raytracing_data.update_view_state(self.view_state_when_pressed, m)
        elif self.mouse_mode == 'rotate':
            RF = self.raytracing_data.RF
            delta_x = event.x - self.mouse_pos_when_pressed[0]
            delta_y = event.y - self.mouse_pos_when_pressed[1]
            angle_x = RF(-delta_x * 0.01)
            angle_y = RF(-delta_y * 0.01)
            m = O13_y_rotation(angle_x) * O13_x_rotation(angle_y)
            self.view_state = self.raytracing_data.update_view_state(self.view_state, m)
            self.mouse_pos_when_pressed = (event.x, event.y)
        else:
            return
        self.redraw_if_initialized()

    def tkButtonRelease1(self, event):
        self.mouse_mode = None
        self.configure(cursor=self.cursor)

    def setup_keymapping(self, keyboard='QWERTY'):
        self.keymapping = _keymappings[keyboard]
        self.key_to_last_accounted_and_release_time = {k: [None, None] for k in self.keymapping}

    def apply_settings(self, settings):
        self.setup_keymapping(settings.get('keyboard', 'QWERTY'))