from kivy.utils import platform
def _on_keyboard_handler(instance, key, scancode, codepoint, modifiers):
    if key == 293 and modifiers == []:
        instance.screenshot()
    elif key == 292 and modifiers == []:
        instance.rotation += 90
    elif key == 292 and modifiers == ['shift']:
        if platform in ('win', 'linux', 'macosx'):
            instance.rotation = 0
            w, h = instance.size
            w, h = (h, w)
            instance.size = (w, h)