def _android_ask_redraw(*largs):
    global g_android_redraw_count
    from kivy.core.window import Window
    Window.canvas.ask_update()
    g_android_redraw_count -= 1
    if g_android_redraw_count < 0:
        return False