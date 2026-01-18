import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.utils.numpy import softmax
Initializes a ParametricRecSys instance.

        Args:
            embedding_size: Embedding size for both users and docs.
                Each value in the user/doc embeddings can have values between
                -1.0 and 1.0.
            num_docs_to_select_from: The number of documents to present to the
                agent each timestep. The agent will then have to pick a slate
                out of these.
            slate_size: The size of the slate to recommend to the user at each
                timestep.
            num_docs_in_db: The total number of documents in the DB. Set this
                to None, in case you would like to resample docs from an
                infinite pool.
            num_users_in_db: The total number of users in the DB. Set this to
                None, in case you would like to resample users from an infinite
                pool.
            user_time_budget: The total time budget a user has throughout an
                episode. Once this time budget is used up (through engagements
                with clicked/selected documents), the episode ends.
        