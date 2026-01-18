from recsim import choice_model
from recsim.environments import (
from ray.rllib.env.wrappers.recsim import make_recsim_env
from ray.tune import register_env
class SingleClusterIEvVideo(iev.IEvVideo):

    def __init__(self, doc_id, features, video_length=None, quality=None):
        super(SingleClusterIEvVideo, self).__init__(doc_id=doc_id, features=features, cluster_id=0, video_length=video_length, quality=quality)