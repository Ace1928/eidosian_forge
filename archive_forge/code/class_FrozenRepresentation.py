import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
class FrozenRepresentation:
    """
    Frozen Statespace Model

    Takes a snapshot of a Statespace model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite
        covariance matrix describing the shocks in the
        measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the shapes of each of
        the representation matrices as tuples.
    endog : ndarray
        The observation vector.
    design : ndarray
        The design matrix, :math:`Z`.
    obs_intercept : ndarray
        The intercept for the observation equation, :math:`d`.
    obs_cov : ndarray
        The covariance matrix for the observation equation :math:`H`.
    transition : ndarray
        The transition matrix, :math:`T`.
    state_intercept : ndarray
        The intercept for the transition equation, :math:`c`.
    selection : ndarray
        The selection matrix, :math:`R`.
    state_cov : ndarray
        The covariance matrix for the state equation :math:`Q`.
    missing : array of bool
        An array of the same size as `endog`, filled
        with boolean values that are True if the
        corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry
        is the number (between 0 and `k_endog`) of NaNs in
        the ith row of the `endog` array.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    initialization : Initialization object
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    """
    _model_attributes = ['model', 'prefix', 'dtype', 'nobs', 'k_endog', 'k_states', 'k_posdef', 'time_invariant', 'endog', 'design', 'obs_intercept', 'obs_cov', 'transition', 'state_intercept', 'selection', 'state_cov', 'missing', 'nmissing', 'shapes', 'initialization', 'initial_state', 'initial_state_cov', 'initial_variance']
    _attributes = _model_attributes

    def __init__(self, model):
        for name in self._attributes:
            setattr(self, name, None)
        self.update_representation(model)

    def update_representation(self, model):
        """Update model Representation"""
        self.model = model
        self.prefix = model.prefix
        self.dtype = model.dtype
        self.nobs = model.nobs
        self.k_endog = model.k_endog
        self.k_states = model.k_states
        self.k_posdef = model.k_posdef
        self.time_invariant = model.time_invariant
        self.endog = model.endog
        self.design = model._design.copy()
        self.obs_intercept = model._obs_intercept.copy()
        self.obs_cov = model._obs_cov.copy()
        self.transition = model._transition.copy()
        self.state_intercept = model._state_intercept.copy()
        self.selection = model._selection.copy()
        self.state_cov = model._state_cov.copy()
        self.missing = np.array(model._statespaces[self.prefix].missing, copy=True)
        self.nmissing = np.array(model._statespaces[self.prefix].nmissing, copy=True)
        self.shapes = dict(model.shapes)
        for name in self.shapes.keys():
            if name == 'obs':
                continue
            self.shapes[name] = getattr(self, name).shape
        self.shapes['obs'] = self.endog.shape
        self.initialization = model.initialization
        if model.initialization is not None:
            model._initialize_state()
            self.initial_state = np.array(model._statespaces[self.prefix].initial_state, copy=True)
            self.initial_state_cov = np.array(model._statespaces[self.prefix].initial_state_cov, copy=True)
            self.initial_diffuse_state_cov = np.array(model._statespaces[self.prefix].initial_diffuse_state_cov, copy=True)