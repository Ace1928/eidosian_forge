import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import (
class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    manager_class = _api.classproperty(lambda cls: FigureManagerWebAgg)
    _timer_cls = TimerAsyncio
    supports_blit = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._png_is_old = True
        self._force_full = True
        self._last_buff = np.empty((0, 0))
        self._current_image_mode = 'full'
        self._last_mouse_xy = (None, None)

    def show(self):
        from matplotlib.pyplot import show
        show()

    def draw(self):
        self._png_is_old = True
        try:
            super().draw()
        finally:
            self.manager.refresh_all()

    def blit(self, bbox=None):
        self._png_is_old = True
        self.manager.refresh_all()

    def draw_idle(self):
        self.send_event('draw')

    def set_cursor(self, cursor):
        cursor = _api.check_getitem({backend_tools.Cursors.HAND: 'pointer', backend_tools.Cursors.POINTER: 'default', backend_tools.Cursors.SELECT_REGION: 'crosshair', backend_tools.Cursors.MOVE: 'move', backend_tools.Cursors.WAIT: 'wait', backend_tools.Cursors.RESIZE_HORIZONTAL: 'ew-resize', backend_tools.Cursors.RESIZE_VERTICAL: 'ns-resize'}, cursor=cursor)
        self.send_event('cursor', cursor=cursor)

    def set_image_mode(self, mode):
        """
        Set the image mode for any subsequent images which will be sent
        to the clients. The modes may currently be either 'full' or 'diff'.

        Note: diff images may not contain transparency, therefore upon
        draw this mode may be changed if the resulting image has any
        transparent component.
        """
        _api.check_in_list(['full', 'diff'], mode=mode)
        if self._current_image_mode != mode:
            self._current_image_mode = mode
            self.handle_send_image_mode(None)

    def get_diff_image(self):
        if self._png_is_old:
            renderer = self.get_renderer()
            pixels = np.asarray(renderer.buffer_rgba())
            buff = pixels.view(np.uint32).squeeze(2)
            if self._force_full or buff.shape != self._last_buff.shape or (pixels[:, :, 3] != 255).any():
                self.set_image_mode('full')
                output = buff
            else:
                self.set_image_mode('diff')
                diff = buff != self._last_buff
                output = np.where(diff, buff, 0)
            self._last_buff = buff.copy()
            self._force_full = False
            self._png_is_old = False
            data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
            with BytesIO() as png:
                Image.fromarray(data).save(png, format='png')
                return png.getvalue()

    def handle_event(self, event):
        e_type = event['type']
        handler = getattr(self, f'handle_{e_type}', self.handle_unknown_event)
        return handler(event)

    def handle_unknown_event(self, event):
        _log.warning('Unhandled message type %s. %s', event['type'], event)

    def handle_ack(self, event):
        pass

    def handle_draw(self, event):
        self.draw()

    def _handle_mouse(self, event):
        x = event['x']
        y = event['y']
        y = self.get_renderer().height - y
        self._last_mouse_xy = (x, y)
        button = event['button'] + 1
        e_type = event['type']
        modifiers = event['modifiers']
        guiEvent = event.get('guiEvent')
        if e_type in ['button_press', 'button_release']:
            MouseEvent(e_type + '_event', self, x, y, button, modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type == 'dblclick':
            MouseEvent('button_press_event', self, x, y, button, dblclick=True, modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type == 'scroll':
            MouseEvent('scroll_event', self, x, y, step=event['step'], modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type == 'motion_notify':
            MouseEvent(e_type + '_event', self, x, y, modifiers=modifiers, guiEvent=guiEvent)._process()
        elif e_type in ['figure_enter', 'figure_leave']:
            LocationEvent(e_type + '_event', self, x, y, modifiers=modifiers, guiEvent=guiEvent)._process()
    handle_button_press = handle_button_release = handle_dblclick = handle_figure_enter = handle_figure_leave = handle_motion_notify = handle_scroll = _handle_mouse

    def _handle_key(self, event):
        KeyEvent(event['type'] + '_event', self, _handle_key(event['key']), *self._last_mouse_xy, guiEvent=event.get('guiEvent'))._process()
    handle_key_press = handle_key_release = _handle_key

    def handle_toolbar_button(self, event):
        getattr(self.toolbar, event['name'])()

    def handle_refresh(self, event):
        figure_label = self.figure.get_label()
        if not figure_label:
            figure_label = f'Figure {self.manager.num}'
        self.send_event('figure_label', label=figure_label)
        self._force_full = True
        if self.toolbar:
            self.toolbar.set_history_buttons()
        self.draw_idle()

    def handle_resize(self, event):
        x = int(event.get('width', 800)) * self.device_pixel_ratio
        y = int(event.get('height', 800)) * self.device_pixel_ratio
        fig = self.figure
        fig.set_size_inches(x / fig.dpi, y / fig.dpi, forward=False)
        self._png_is_old = True
        self.manager.resize(*fig.bbox.size, forward=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    def handle_send_image_mode(self, event):
        self.send_event('image_mode', mode=self._current_image_mode)

    def handle_set_device_pixel_ratio(self, event):
        self._handle_set_device_pixel_ratio(event.get('device_pixel_ratio', 1))

    def handle_set_dpi_ratio(self, event):
        self._handle_set_device_pixel_ratio(event.get('dpi_ratio', 1))

    def _handle_set_device_pixel_ratio(self, device_pixel_ratio):
        if self._set_device_pixel_ratio(device_pixel_ratio):
            self._force_full = True
            self.draw_idle()

    def send_event(self, event_type, **kwargs):
        if self.manager:
            self.manager._send_event(event_type, **kwargs)