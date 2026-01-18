import enum
import functools
from typing import Optional
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import (
class Catalog:
    """Describes the sub-module-architectures to be used in RLModules.

    RLlib's native RLModules get their Models from a Catalog object.
    By default, that Catalog builds the configs it has as attributes.
    This component was build to be hackable and extensible. You can inject custom
    components into RL Modules by overriding the `build_xxx` methods of this class.
    Note that it is recommended to write a custom RL Module for a single use-case.
    Modifications to Catalogs mostly make sense if you want to reuse the same
    Catalog for different RL Modules. For example if you have written a custom
    encoder and want to inject it into different RL Modules (e.g. for PPO, DQN, etc.).
    You can influence the decision tree that determines the sub-components by modifying
    `Catalog._determine_components_hook`.

    Usage example:

    # Define a custom catalog

    .. testcode::

        import torch
        import gymnasium as gym
        from ray.rllib.core.models.configs import MLPHeadConfig
        from ray.rllib.core.models.catalog import Catalog


        class MyCatalog(Catalog):
            def __init__(
                self,
                observation_space: gym.Space,
                action_space: gym.Space,
                model_config_dict: dict,
            ):
                super().__init__(observation_space, action_space, model_config_dict)
                self.my_model_config_dict = MLPHeadConfig(
                    hidden_layer_dims=[64, 32],
                    input_dims=[self.observation_space.shape[0]],
                )

            def build_my_head(self, framework: str):
                return self.my_model_config_dict.build(framework=framework)

        # With that, RLlib can build and use models from this catalog like this:
        catalog = MyCatalog(gym.spaces.Box(0, 1), gym.spaces.Box(0, 1), {})
        my_head = catalog.build_my_head("torch")
        out = my_head(torch.Tensor([[1]]))
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: dict, view_requirements: dict=None):
        """Initializes a Catalog with a default encoder config.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            model_config_dict: The model config that specifies things like hidden
                dimensions and activations functions to use in this Catalog.
            view_requirements: The view requirements of Models to produce. This is
                needed for a Model that encodes a complex temporal mix of
                observations, actions or rewards.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self._model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}
        self._view_requirements = view_requirements
        self._latent_dims = None
        self._determine_components_hook()

    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def _determine_components_hook(self):
        """Decision tree hook for subclasses to override.

        By default, this method executes the decision tree that determines the
        components that a Catalog builds. You can extend the components by overriding
        this or by adding to the constructor of your subclass.

        Override this method if you don't want to use the default components
        determined here. If you want to use them but add additional components, you
        should call `super()._determine_components()` at the beginning of your
        implementation.

        This makes it so that subclasses are not forced to create an encoder config
        if the rest of their catalog is not dependent on it or if it breaks.
        At the end of this method, an attribute `Catalog.latent_dims`
        should be set so that heads can be built using that information.
        """
        self._encoder_config = self._get_encoder_config(observation_space=self.observation_space, action_space=self.action_space, model_config_dict=self._model_config_dict, view_requirements=self._view_requirements)
        self._action_dist_class_fn = functools.partial(self._get_dist_cls_from_action_space, action_space=self.action_space)
        self.latent_dims = self._encoder_config.output_dims

    @property
    def latent_dims(self):
        """Returns the latent dimensions of the encoder.

        This establishes an agreement between encoder and heads about the latent
        dimensions. Encoders can be built to output a latent tensor with
        `latent_dims` dimensions, and heads can be built with tensors of
        `latent_dims` dimensions as inputs. This can be safely ignored if this
        agreement is not needed in case of modifications to the Catalog.

        Returns:
            The latent dimensions of the encoder.
        """
        return self._latent_dims

    @latent_dims.setter
    def latent_dims(self, value):
        self._latent_dims = value

    @OverrideToImplementCustomLogic
    def build_encoder(self, framework: str) -> Encoder:
        """Builds the encoder.

        By default, this method builds an encoder instance from Catalog._encoder_config.

        You should override this if you want to use RLlib's default RL Modules but
        only want to change the encoder. For example, if you want to use a custom
        encoder, but want to use RLlib's default heads, action distribution and how
        tensors are routed between them. If you want to have full control over the
        RL Module, we recommend writing your own RL Module by inheriting from one of
        RLlib's RL Modules instead.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The encoder.
        """
        assert hasattr(self, '_encoder_config'), 'You must define a `Catalog._encoder_config` attribute in your Catalog subclass or override the `Catalog.build_encoder` method. By default, an encoder_config is created in the __post_init__ method.'
        return self._encoder_config.build(framework=framework)

    @OverrideToImplementCustomLogic
    def get_action_dist_cls(self, framework: str):
        """Get the action distribution class.

        The default behavior is to get the action distribution from the
        `Catalog._action_dist_class_fn`.

        You should override this to have RLlib build your custom action
        distribution instead of the default one. For example, if you don't want to
        use RLlib's default RLModules with their default models, but only want to
        change the distribution that Catalog returns.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The action distribution.
        """
        assert hasattr(self, '_action_dist_class_fn'), 'You must define a `Catalog._action_dist_class_fn` attribute in your Catalog subclass or override the `Catalog.action_dist_class_fn` method. By default, an action_dist_class_fn is created in the __post_init__ method.'
        return self._action_dist_class_fn(framework=framework)

    @classmethod
    def _get_encoder_config(cls, observation_space: gym.Space, model_config_dict: dict, action_space: gym.Space=None, view_requirements=None) -> ModelConfig:
        """Returns an EncoderConfig for the given input_space and model_config_dict.

        Encoders are usually used in RLModules to transform the input space into a
        latent space that is then fed to the heads. The returned EncoderConfig
        objects correspond to the built-in Encoder classes in RLlib.
        For example, for a simple 1D-Box input_space, RLlib offers an
        MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
        this method to produce specific EncoderConfigs for your custom Models.

        The following input spaces lead to the following configs:
        - 1D-Box: MLPEncoderConfig
        - 3D-Box: CNNEncoderConfig
        # TODO (Artur): Support more spaces here
        # ...

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            action_space: The action space to use if actions are to be encoded. This
                is commonly the case for LSTM models.
            view_requirements: The view requirements to use if anything else than
                observation_space or action_space is to be encoded. This signifies an
                advanced use case.

        Returns:
            The encoder config.
        """
        model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}
        activation = model_config_dict['fcnet_activation']
        output_activation = model_config_dict['fcnet_activation']
        fcnet_hiddens = model_config_dict['fcnet_hiddens']
        encoder_latent_dim = model_config_dict['encoder_latent_dim'] or fcnet_hiddens[-1]
        use_lstm = model_config_dict['use_lstm']
        use_attention = model_config_dict['use_attention']
        if use_lstm:
            encoder_config = RecurrentEncoderConfig(input_dims=observation_space.shape, recurrent_layer_type='lstm', hidden_dim=model_config_dict['lstm_cell_size'], batch_major=not model_config_dict['_time_major'], num_layers=1, tokenizer_config=cls.get_tokenizer_config(observation_space, model_config_dict, view_requirements))
        elif use_attention:
            raise NotImplementedError
        elif isinstance(observation_space, Box) and len(observation_space.shape) == 1:
            if model_config_dict['encoder_latent_dim']:
                hidden_layer_dims = model_config_dict['fcnet_hiddens']
            else:
                hidden_layer_dims = model_config_dict['fcnet_hiddens'][:-1]
            encoder_config = MLPEncoderConfig(input_dims=observation_space.shape, hidden_layer_dims=hidden_layer_dims, hidden_layer_activation=activation, output_layer_dim=encoder_latent_dim, output_layer_activation=output_activation)
        elif isinstance(observation_space, Box) and len(observation_space.shape) == 3:
            if not model_config_dict.get('conv_filters'):
                model_config_dict['conv_filters'] = get_filter_config(observation_space.shape)
            encoder_config = CNNEncoderConfig(input_dims=observation_space.shape, cnn_filter_specifiers=model_config_dict['conv_filters'], cnn_activation=model_config_dict['conv_activation'], cnn_use_layernorm=model_config_dict.get('conv_use_layernorm', False))
        elif isinstance(observation_space, Box) and len(observation_space.shape) == 2:
            raise ValueError(f'No default encoder config for obs space={observation_space}, lstm={use_lstm} and attention={use_attention} found. 2D Box spaces are not supported. They should be either flattened to a 1D Box space or enhanced to be a 3D box space.')
        else:
            raise ValueError(f'No default encoder config for obs space={observation_space}, lstm={use_lstm} and attention={use_attention} found.')
        return encoder_config

    @classmethod
    @OverrideToImplementCustomLogic
    def get_tokenizer_config(cls, observation_space: gym.Space, model_config_dict: dict, view_requirements: Optional[ViewRequirementsDict]=None) -> ModelConfig:
        """Returns a tokenizer config for the given space.

        This is useful for recurrent / tranformer models that need to tokenize their
        inputs. By default, RLlib uses the models supported by Catalog out of the box to
        tokenize.

        You should override this method if you want to change the custom tokenizer
        inside current encoders that Catalog returns without providing the recurrent
        network as a whole. For example, if you want to define some custom CNN layers
        as a tokenizer for a recurrent encoder that already includes the recurrent
        layers and handles the state.

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            view_requirements: The view requirements to use if anything else than
                observation_space is to be encoded. This signifies an advanced use case.
        """
        return cls._get_encoder_config(observation_space=observation_space, model_config_dict={**model_config_dict, **{'use_lstm': False, 'use_attention': False}}, view_requirements=view_requirements)

    @classmethod
    def _get_dist_cls_from_action_space(cls, action_space: gym.Space, *, framework: Optional[str]=None) -> Distribution:
        """Returns a distribution class for the given action space.

        You can get the required input dimension for the distribution by calling
        `action_dict_cls.required_input_dim(action_space)`
        on the retrieved class. This is useful, because the Catalog needs to find out
        about the required input dimension for the distribution before the model that
        outputs these inputs is configured.

        Args:
            action_space: Action space of the target gym env.
            framework: The framework to use.

        Returns:
            The distribution class for the given action space.
        """

        class DistEnum(enum.Enum):
            Categorical = 'Categorical'
            DiagGaussian = 'Gaussian'
            Deterministic = 'Deterministic'
            MultiDistribution = 'MultiDistribution'
            MultiCategorical = 'MultiCategorical'
        if framework == 'torch':
            from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDeterministic, TorchDiagGaussian
            distribution_dicts = {DistEnum.Deterministic: TorchDeterministic, DistEnum.DiagGaussian: TorchDiagGaussian, DistEnum.Categorical: TorchCategorical}
        elif framework == 'tf2':
            from ray.rllib.models.tf.tf_distributions import TfCategorical, TfDeterministic, TfDiagGaussian
            distribution_dicts = {DistEnum.Deterministic: TfDeterministic, DistEnum.DiagGaussian: TfDiagGaussian, DistEnum.Categorical: TfCategorical}
        else:
            raise ValueError(f"Unknown framework: {framework}. Only 'torch' and 'tf2' are supported for RLModule Catalogs.")
        if isinstance(action_space, (Tuple, Dict)):
            partial_multi_action_distribution_cls = _multi_action_dist_partial_helper(catalog_cls=cls, action_space=action_space, framework=framework)
            distribution_dicts[DistEnum.MultiDistribution] = partial_multi_action_distribution_cls
        if isinstance(action_space, MultiDiscrete):
            partial_multi_categorical_distribution_cls = _multi_categorical_dist_partial_helper(action_space=action_space, framework=framework)
            distribution_dicts[DistEnum.MultiCategorical] = partial_multi_categorical_distribution_cls
        if isinstance(action_space, Box):
            if action_space.dtype.char in np.typecodes['AllInteger']:
                raise ValueError('Box(..., `int`) action spaces are not supported. Use MultiDiscrete  or Box(..., `float`).')
            else:
                if len(action_space.shape) > 1:
                    raise UnsupportedSpaceException(f'Action space has multiple dimensions {action_space.shape}. Consider reshaping this into a single dimension, using a custom action distribution, using a Tuple action space, or the multi-agent API.')
                return distribution_dicts[DistEnum.DiagGaussian]
        elif isinstance(action_space, Discrete):
            return distribution_dicts[DistEnum.Categorical]
        elif isinstance(action_space, (Tuple, Dict)):
            return distribution_dicts[DistEnum.MultiDistribution]
        elif isinstance(action_space, Simplex):
            raise NotImplementedError('Simplex action space not yet supported.')
        elif isinstance(action_space, MultiDiscrete):
            return distribution_dicts[DistEnum.MultiCategorical]
        else:
            raise NotImplementedError(f'Unsupported action space: `{action_space}`')

    @staticmethod
    def get_preprocessor(observation_space: gym.Space, **kwargs) -> Preprocessor:
        """Returns a suitable preprocessor for the given observation space.

        Args:
            observation_space: The input observation space.
            **kwargs: Forward-compatible kwargs.

        Returns:
            preprocessor: Preprocessor for the observations.
        """
        options = kwargs.get('options', {})
        if options:
            deprecation_warning(old='get_preprocessor_for_space(..., options={...})', help='Override `Catalog.get_preprocessor()` in order to implement custom behaviour.', error=False)
        if options.get('custom_preprocessor'):
            deprecation_warning(old="model_config['custom_preprocessor']", help='Custom preprocessors are deprecated, since they sometimes conflict with the built-in preprocessors for handling complex observation spaces. Please use wrapper classes around your environment instead.', error=True)
        else:
            cls = get_preprocessor(observation_space)
            prep = cls(observation_space, options)
            return prep