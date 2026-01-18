import numpy as np
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.VideoClip import ColorClip, VideoClip
class CompositeVideoClip(VideoClip):
    """ 
    
    A VideoClip made of other videoclips displayed together. This is the
    base class for most compositions.
    
    Parameters
    ----------

    size
      The size (height x width) of the final clip.

    clips
      A list of videoclips. Each clip of the list will
      be displayed below the clips appearing after it in the list.
      For each clip:
       
      - The attribute ``pos`` determines where the clip is placed.
          See ``VideoClip.set_pos``
      - The mask of the clip determines which parts are visible.
        
      Finally, if all the clips in the list have their ``duration``
      attribute set, then the duration of the composite video clip
      is computed automatically

    bg_color
      Color for the unmasked and unfilled regions. Set to None for these
      regions to be transparent (will be slower).

    use_bgclip
      Set to True if the first clip in the list should be used as the
      'background' on which all other clips are blitted. That first clip must
      have the same size as the final clip. If it has no transparency, the final
      clip will have no mask. 
    
    The clip with the highest FPS will be the FPS of the composite clip.

    """

    def __init__(self, clips, size=None, bg_color=None, use_bgclip=False, ismask=False):
        if size is None:
            size = clips[0].size
        if use_bgclip and clips[0].mask is None:
            transparent = False
        else:
            transparent = bg_color is None
        if bg_color is None:
            bg_color = 0.0 if ismask else (0, 0, 0)
        fpss = [c.fps for c in clips if getattr(c, 'fps', None)]
        self.fps = max(fpss) if fpss else None
        VideoClip.__init__(self)
        self.size = size
        self.ismask = ismask
        self.clips = clips
        self.bg_color = bg_color
        if use_bgclip:
            self.bg = clips[0]
            self.clips = clips[1:]
            self.created_bg = False
        else:
            self.clips = clips
            self.bg = ColorClip(size, color=self.bg_color)
            self.created_bg = True
        ends = [c.end for c in self.clips]
        if None not in ends:
            duration = max(ends)
            self.duration = duration
            self.end = duration
        audioclips = [v.audio for v in self.clips if v.audio is not None]
        if audioclips:
            self.audio = CompositeAudioClip(audioclips)
        if transparent:
            maskclips = [(c.mask if c.mask is not None else c.add_mask().mask).set_position(c.pos).set_end(c.end).set_start(c.start, change_end=False) for c in self.clips]
            self.mask = CompositeVideoClip(maskclips, self.size, ismask=True, bg_color=0.0)

        def make_frame(t):
            """ The clips playing at time `t` are blitted over one
                another. """
            f = self.bg.get_frame(t)
            for c in self.playing_clips(t):
                f = c.blit_on(f, t)
            return f
        self.make_frame = make_frame

    def playing_clips(self, t=0):
        """ Returns a list of the clips in the composite clips that are
            actually playing at the given time `t`. """
        return [c for c in self.clips if c.is_playing(t)]

    def close(self):
        if self.created_bg and self.bg:
            self.bg.close()
            self.bg = None
        if hasattr(self, 'audio') and self.audio:
            self.audio.close()
            self.audio = None