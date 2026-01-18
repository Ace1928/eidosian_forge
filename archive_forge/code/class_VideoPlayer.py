from json import load
from os.path import exists
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty, \
from kivy.animation import Animation
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.video import Image
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.clock import Clock
class VideoPlayer(GridLayout):
    """VideoPlayer class. See module documentation for more information.
    """
    source = StringProperty('')
    "Source of the video to read.\n\n    :attr:`source` is a :class:`~kivy.properties.StringProperty` and\n    defaults to ''.\n\n    .. versionchanged:: 1.4.0\n    "
    thumbnail = StringProperty('')
    "Thumbnail of the video to show. If None, VideoPlayer will try to find\n    the thumbnail from the :attr:`source` + '.png'.\n\n    :attr:`thumbnail` a :class:`~kivy.properties.StringProperty` and defaults\n    to ''.\n\n    .. versionchanged:: 1.4.0\n    "
    duration = NumericProperty(-1)
    'Duration of the video. The duration defaults to -1 and is set to the\n    real duration when the video is loaded.\n\n    :attr:`duration` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to -1.\n    '
    position = NumericProperty(0)
    'Position of the video between 0 and :attr:`duration`. The position\n    defaults to -1 and is set to the real position when the video is loaded.\n\n    :attr:`position` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to -1.\n    '
    volume = NumericProperty(1.0)
    'Volume of the video in the range 0-1. 1 means full volume and 0 means\n    mute.\n\n    :attr:`volume` is a :class:`~kivy.properties.NumericProperty` and defaults\n    to 1.\n    '
    state = OptionProperty('stop', options=('play', 'pause', 'stop'))
    "String, indicates whether to play, pause, or stop the video::\n\n        # start playing the video at creation\n        video = VideoPlayer(source='movie.mkv', state='play')\n\n        # create the video, and start later\n        video = VideoPlayer(source='movie.mkv')\n        # and later\n        video.state = 'play'\n\n    :attr:`state` is an :class:`~kivy.properties.OptionProperty` and defaults\n    to 'stop'.\n    "
    play = BooleanProperty(False, deprecated=True)
    "\n    .. deprecated:: 1.4.0\n        Use :attr:`state` instead.\n\n    Boolean, indicates whether the video is playing or not. You can start/stop\n    the video by setting this property::\n\n        # start playing the video at creation\n        video = VideoPlayer(source='movie.mkv', play=True)\n\n        # create the video, and start later\n        video = VideoPlayer(source='movie.mkv')\n        # and later\n        video.play = True\n\n    :attr:`play` is a :class:`~kivy.properties.BooleanProperty` and defaults\n    to False.\n    "
    image_overlay_play = StringProperty('atlas://data/images/defaulttheme/player-play-overlay')
    'Image filename used to show a "play" overlay when the video has not yet\n    started.\n\n    :attr:`image_overlay_play` is a\n    :class:`~kivy.properties.StringProperty` and\n    defaults to \'atlas://data/images/defaulttheme/player-play-overlay\'.\n\n    '
    image_loading = StringProperty('data/images/image-loading.zip')
    "Image filename used when the video is loading.\n\n    :attr:`image_loading` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'data/images/image-loading.zip'.\n    "
    image_play = StringProperty('atlas://data/images/defaulttheme/media-playback-start')
    'Image filename used for the "Play" button.\n\n    :attr:`image_play` is a :class:`~kivy.properties.StringProperty` and\n    defaults to \'atlas://data/images/defaulttheme/media-playback-start\'.\n    '
    image_stop = StringProperty('atlas://data/images/defaulttheme/media-playback-stop')
    'Image filename used for the "Stop" button.\n\n    :attr:`image_stop` is a :class:`~kivy.properties.StringProperty` and\n    defaults to \'atlas://data/images/defaulttheme/media-playback-stop\'.\n    '
    image_pause = StringProperty('atlas://data/images/defaulttheme/media-playback-pause')
    'Image filename used for the "Pause" button.\n\n    :attr:`image_pause` is a :class:`~kivy.properties.StringProperty` and\n    defaults to \'atlas://data/images/defaulttheme/media-playback-pause\'.\n    '
    image_volumehigh = StringProperty('atlas://data/images/defaulttheme/audio-volume-high')
    "Image filename used for the volume icon when the volume is high.\n\n    :attr:`image_volumehigh` is a :class:`~kivy.properties.StringProperty` and\n    defaults to 'atlas://data/images/defaulttheme/audio-volume-high'.\n    "
    image_volumemedium = StringProperty('atlas://data/images/defaulttheme/audio-volume-medium')
    "Image filename used for the volume icon when the volume is medium.\n\n    :attr:`image_volumemedium` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/audio-volume-medium'.\n    "
    image_volumelow = StringProperty('atlas://data/images/defaulttheme/audio-volume-low')
    "Image filename used for the volume icon when the volume is low.\n\n    :attr:`image_volumelow` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/audio-volume-low'.\n    "
    image_volumemuted = StringProperty('atlas://data/images/defaulttheme/audio-volume-muted')
    "Image filename used for the volume icon when the volume is muted.\n\n    :attr:`image_volumemuted` is a :class:`~kivy.properties.StringProperty`\n    and defaults to 'atlas://data/images/defaulttheme/audio-volume-muted'.\n    "
    annotations = StringProperty('')
    "If set, it will be used for reading annotations box.\n\n    :attr:`annotations` is a :class:`~kivy.properties.StringProperty`\n    and defaults to ''.\n    "
    fullscreen = BooleanProperty(False)
    "Switch to fullscreen view. This should be used with care. When\n    activated, the widget will remove itself from its parent, remove all\n    children from the window and will add itself to it. When fullscreen is\n    unset, all the previous children are restored and the widget is restored to\n    its previous parent.\n\n    .. warning::\n\n        The re-add operation doesn't care about the index position of its\n        children within the parent.\n\n    :attr:`fullscreen` is a :class:`~kivy.properties.BooleanProperty`\n    and defaults to False.\n    "
    allow_fullscreen = BooleanProperty(True)
    'By default, you can double-tap on the video to make it fullscreen. Set\n    this property to False to prevent this behavior.\n\n    :attr:`allow_fullscreen` is a :class:`~kivy.properties.BooleanProperty`\n    defaults to True.\n    '
    options = DictProperty({})
    'Optional parameters can be passed to a :class:`~kivy.uix.video.Video`\n    instance with this property.\n\n    :attr:`options` a :class:`~kivy.properties.DictProperty` and\n    defaults to {}.\n    '
    container = ObjectProperty(None)
    _video_load_ev = None

    def __init__(self, **kwargs):
        self._video = None
        self._image = None
        self._annotations = ''
        self._annotations_labels = []
        super(VideoPlayer, self).__init__(**kwargs)
        update_thumbnail = self._update_thumbnail
        update_annotations = self._update_annotations
        fbind = self.fbind
        fbind('thumbnail', update_thumbnail)
        fbind('annotations', update_annotations)
        if self.source:
            self._trigger_video_load()

    def _trigger_video_load(self, *largs):
        ev = self._video_load_ev
        if ev is None:
            ev = self._video_load_ev = Clock.schedule_once(self._do_video_load, -1)
        ev()

    def _try_load_default_thumbnail(self, *largs):
        if not self.thumbnail:
            filename = self.source.rsplit('.', 1)
            thumbnail = filename[0] + '.png'
            if exists(thumbnail):
                self._load_thumbnail(thumbnail)

    def _try_load_default_annotations(self, *largs):
        if not self.annotations:
            filename = self.source.rsplit('.', 1)
            annotations = filename[0] + '.jsa'
            if exists(annotations):
                self._load_annotations(annotations)

    def on_source(self, instance, value):
        Clock.schedule_once(self._try_load_default_thumbnail, -1)
        Clock.schedule_once(self._try_load_default_annotations, -1)
        if self._video is not None:
            self._video.unload()
            self._video = None
        if value:
            self._trigger_video_load()

    def _update_thumbnail(self, *largs):
        self._load_thumbnail(self.thumbnail)

    def _update_annotations(self, *largs):
        self._load_annotations(self.annotations)

    def on_image_overlay_play(self, instance, value):
        self._image.image_overlay_play = value

    def on_image_loading(self, instance, value):
        self._image.image_loading = value

    def _load_thumbnail(self, thumbnail):
        if not self.container:
            return
        self.container.clear_widgets()
        if thumbnail:
            self._image = VideoPlayerPreview(source=thumbnail, video=self)
            self.container.add_widget(self._image)

    def _load_annotations(self, annotations):
        if not self.container:
            return
        self._annotations_labels = []
        if annotations:
            with open(annotations, 'r') as fd:
                self._annotations = load(fd)
            if self._annotations:
                for ann in self._annotations:
                    self._annotations_labels.append(VideoPlayerAnnotation(annotation=ann))

    def on_state(self, instance, value):
        if self._video is not None:
            self._video.state = value

    def _set_state(self, instance, value):
        self.state = value

    def _do_video_load(self, *largs):
        self._video = Video(source=self.source, state=self.state, volume=self.volume, pos_hint={'x': 0, 'y': 0}, **self.options)
        self._video.bind(texture=self._play_started, duration=self.setter('duration'), position=self.setter('position'), volume=self.setter('volume'), state=self._set_state)

    def on_play(self, instance, value):
        value = 'play' if value else 'stop'
        return self.on_state(instance, value)

    def on_volume(self, instance, value):
        if not self._video:
            return
        self._video.volume = value

    def on_position(self, instance, value):
        labels = self._annotations_labels
        if not labels:
            return
        for label in labels:
            start = label.start
            duration = label.duration
            if start > value or start + duration < value:
                if label.parent:
                    label.parent.remove_widget(label)
            elif label.parent is None:
                self.container.add_widget(label)

    def seek(self, percent, precise=True):
        """Change the position to a percentage (strictly, a proportion)
           of duration.

        :Parameters:
            `percent`: float or int
                Position to seek as a proportion of total duration, must
                be between 0-1.
            `precise`: bool, defaults to True
                Precise seeking is slower, but seeks to exact requested
                percent.

        .. warning::
            Calling seek() before the video is loaded has no effect.

        .. versionadded:: 1.2.0

        .. versionchanged:: 1.10.1
            The `precise` keyword argument has been added.
        """
        if not self._video:
            return
        self._video.seek(percent, precise=precise)

    def _play_started(self, instance, value):
        self.container.clear_widgets()
        self.container.add_widget(self._video)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        if touch.is_double_tap and self.allow_fullscreen:
            self.fullscreen = not self.fullscreen
            return True
        return super(VideoPlayer, self).on_touch_down(touch)

    def on_fullscreen(self, instance, value):
        window = self.get_parent_window()
        if not window:
            Logger.warning('VideoPlayer: Cannot switch to fullscreen, window not found.')
            if value:
                self.fullscreen = False
            return
        if not self.parent:
            Logger.warning('VideoPlayer: Cannot switch to fullscreen, no parent.')
            if value:
                self.fullscreen = False
            return
        if value:
            self._fullscreen_state = state = {'parent': self.parent, 'pos': self.pos, 'size': self.size, 'pos_hint': self.pos_hint, 'size_hint': self.size_hint, 'window_children': window.children[:]}
            for child in window.children[:]:
                window.remove_widget(child)
            if state['parent'] is not window:
                state['parent'].remove_widget(self)
            window.add_widget(self)
            self.pos = (0, 0)
            self.size = (100, 100)
            self.pos_hint = {}
            self.size_hint = (1, 1)
        else:
            state = self._fullscreen_state
            window.remove_widget(self)
            for child in state['window_children']:
                window.add_widget(child)
            self.pos_hint = state['pos_hint']
            self.size_hint = state['size_hint']
            self.pos = state['pos']
            self.size = state['size']
            if state['parent'] is not window:
                state['parent'].add_widget(self)