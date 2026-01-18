from moviepy.decorators import requires_duration
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.VideoClip import ImageClip
 Momentarily freeze the clip at time t.

    Set `t='end'` to freeze the clip at the end (actually it will freeze on the
    frame at time clip.duration - padding_end seconds).
    With ``duration``you can specify the duration of the freeze.
    With ``total_duration`` you can specify the total duration of
    the clip and the freeze (i.e. the duration of the freeze is
    automatically calculated). One of them must be provided.
    