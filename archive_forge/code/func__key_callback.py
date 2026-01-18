import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _key_callback(self, window, key, scancode, action, mods):
    if action != glfw.RELEASE:
        return
    elif key == glfw.KEY_TAB:
        self.cam.fixedcamid += 1
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        if self.cam.fixedcamid >= self.model.ncam:
            self.cam.fixedcamid = -1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    elif key == glfw.KEY_SPACE and self._paused is not None:
        self._paused = not self._paused
    elif key == glfw.KEY_RIGHT and self._paused is not None:
        self._advance_by_one_step = True
        self._paused = True
    elif key == glfw.KEY_S:
        self._run_speed /= 2.0
    elif key == glfw.KEY_F:
        self._run_speed *= 2.0
    elif key == glfw.KEY_D:
        self._render_every_frame = not self._render_every_frame
    elif key == glfw.KEY_T:
        img = np.zeros((glfw.get_framebuffer_size(self.window)[1], glfw.get_framebuffer_size(self.window)[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, self.viewport, self.con)
        imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
        self._image_idx += 1
    elif key == glfw.KEY_C:
        self._contacts = not self._contacts
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
    elif key == glfw.KEY_E:
        self.vopt.frame = 1 - self.vopt.frame
    elif key == glfw.KEY_H:
        self._hide_menu = not self._hide_menu
    elif key == glfw.KEY_R:
        self._transparent = not self._transparent
        if self._transparent:
            self.model.geom_rgba[:, 3] /= 5.0
        else:
            self.model.geom_rgba[:, 3] *= 5.0
    elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
        self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
    if key == glfw.KEY_ESCAPE:
        print('Pressed ESC')
        print('Quitting.')
        glfw.destroy_window(self.window)
        glfw.terminate()