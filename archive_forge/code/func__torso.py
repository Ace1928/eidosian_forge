import collections
import utils
import tensorflow as tf
def _torso(self, prev_action, env_output):
    reward, _, frame, _, _ = env_output
    frame = tf.cast(frame, tf.float32)
    frame /= 255
    conv_out = frame
    for stack in self._stacks:
        conv_out = stack(conv_out)
    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = self._conv_to_linear(conv_out)
    conv_out = tf.nn.relu(conv_out)
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    return tf.concat([conv_out, clipped_reward, one_hot_prev_action], axis=1)