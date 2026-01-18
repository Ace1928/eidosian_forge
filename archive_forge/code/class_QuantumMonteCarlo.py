import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import QubitUnitary
class QuantumMonteCarlo(Operation):
    """Performs the `quantum Monte Carlo estimation <https://arxiv.org/abs/1805.00109>`__
    algorithm.

    Given a probability distribution :math:`p(i)` of dimension :math:`M = 2^{m}` for some
    :math:`m \\geq 1` and a function :math:`f: X \\rightarrow [0, 1]` defined on the set of
    integers :math:`X = \\{0, 1, \\ldots, M - 1\\}`, this function implements the algorithm that
    allows the following expectation value to be estimated:

    .. math::

        \\mu = \\sum_{i \\in X} p(i) f(i).

    .. figure:: ../../_static/templates/subroutines/qmc.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        probs (array): input probability distribution as a flat array
        func (callable): input function :math:`f` defined on the set of integers
            :math:`X = \\{0, 1, \\ldots, M - 1\\}` such that :math:`f(i)\\in [0, 1]` for :math:`i \\in X`
        target_wires (Union[Wires, Sequence[int], or int]): the target wires
        estimation_wires (Union[Wires, Sequence[int], or int]): the estimation wires

    Raises:
        ValueError: if ``probs`` is not flat or has a length that is not compatible with
            ``target_wires``

    .. note::

        This template is only compatible with simulators because the algorithm is performed using
        unitary matrices. Additionally, this operation is not differentiable. To implement the
        quantum Monte Carlo algorithm on hardware requires breaking down the unitary matrices into
        hardware-compatible gates, check out the :func:`~.quantum_monte_carlo` transformation for
        more details.

    .. details::
        :title: Usage Details

        The algorithm proceeds as follows:

        #. The probability distribution :math:`p(i)` is encoded using a unitary :math:`\\mathcal{A}`
           applied to the first :math:`m` qubits specified by ``target_wires``.
        #. The function :math:`f(i)` is encoded onto the last qubit of ``target_wires`` using a unitary
           :math:`\\mathcal{R}`.
        #. The unitary :math:`\\mathcal{Q}` is defined with eigenvalues
           :math:`e^{\\pm 2 \\pi i \\theta}` such that the phase :math:`\\theta` encodes the expectation
           value through the equation :math:`\\mu = (1 + \\cos (\\pi \\theta)) / 2`. The circuit in steps 1
           and 2 prepares an equal superposition over the two states corresponding to the eigenvalues
           :math:`e^{\\pm 2 \\pi i \\theta}`.
        #. The :func:`~.QuantumPhaseEstimation` circuit is applied so that :math:`\\pm\\theta` can be
           estimated by finding the probabilities of the :math:`n` estimation wires. This in turn allows
           for the estimation of :math:`\\mu`.

        Visit `Rebentrost et al. (2018) <https://arxiv.org/abs/1805.00109>`__ for further details. In
        this algorithm, the number of applications :math:`N` of the :math:`\\mathcal{Q}` unitary scales
        as :math:`2^{n}`. However, due to the use of quantum phase estimation, the error
        :math:`\\epsilon` scales as :math:`\\mathcal{O}(2^{-n})`. Hence,

        .. math::

            N = \\mathcal{O}\\left(\\frac{1}{\\epsilon}\\right).

        This scaling can be compared to standard Monte Carlo estimation, where :math:`N` samples are
        generated from the probability distribution and the average over :math:`f` is taken. In that
        case,

        .. math::

            N =  \\mathcal{O}\\left(\\frac{1}{\\epsilon^{2}}\\right).

        Hence, the quantum Monte Carlo algorithm has a quadratically improved time complexity with
        :math:`N`. An example use case is given below.

        Consider a standard normal distribution :math:`p(x)` and a function
        :math:`f(x) = \\sin ^{2} (x)`. The expectation value of :math:`f(x)` is
        :math:`\\int_{-\\infty}^{\\infty}f(x)p(x)dx \\approx 0.432332`. This number can be approximated by
        discretizing the problem and using the quantum Monte Carlo algorithm.

        First, the problem is discretized:

        .. code-block:: python

            from scipy.stats import norm

            m = 5
            M = 2 ** m

            xmax = np.pi  # bound to region [-pi, pi]
            xs = np.linspace(-xmax, xmax, M)

            probs = np.array([norm().pdf(x) for x in xs])
            probs /= np.sum(probs)

            func = lambda i: np.sin(xs[i]) ** 2

        The ``QuantumMonteCarlo`` template can then be used:

        .. code-block::

            n = 10
            N = 2 ** n

            target_wires = range(m + 1)
            estimation_wires = range(m + 1, n + m + 1)

            dev = qml.device("default.qubit", wires=(n + m + 1))

            @qml.qnode(dev)
            def circuit():
                qml.templates.QuantumMonteCarlo(
                    probs,
                    func,
                    target_wires=target_wires,
                    estimation_wires=estimation_wires,
                )
                return qml.probs(estimation_wires)

            phase_estimated = np.argmax(circuit()[:int(N / 2)]) / N

        The estimated value can be retrieved using the formula :math:`\\mu = (1-\\cos(\\pi \\theta))/2`

        >>> (1 - np.cos(np.pi * phase_estimated)) / 2
        0.4327096457464369
    """
    num_wires = AnyWires
    grad_method = None

    @classmethod
    def _unflatten(cls, data, metadata):
        new_op = cls.__new__(cls)
        new_op._hyperparameters = dict(metadata[1])
        Operation.__init__(new_op, *data, wires=metadata[0])
        return new_op

    def __init__(self, probs, func, target_wires, estimation_wires, id=None):
        if isinstance(probs, np.ndarray) and probs.ndim != 1:
            raise ValueError('The probability distribution must be specified as a flat array')
        dim_p = len(probs)
        num_target_wires_ = np.log2(2 * dim_p)
        num_target_wires = int(num_target_wires_)
        if not np.allclose(num_target_wires_, num_target_wires):
            raise ValueError('The probability distribution must have a length that is a power of two')
        target_wires = list(target_wires)
        estimation_wires = list(estimation_wires)
        wires = target_wires + estimation_wires
        if num_target_wires != len(target_wires):
            raise ValueError(f'The probability distribution of dimension {dim_p} requires {num_target_wires} target wires')
        self._hyperparameters = {'estimation_wires': estimation_wires, 'target_wires': target_wires}
        A = probs_to_unitary(probs)
        R = func_to_unitary(func, dim_p)
        Q = make_Q(A, R)
        super().__init__(A, R, Q, wires=wires, id=id)

    @property
    def num_params(self):
        return 3

    @staticmethod
    def compute_decomposition(A, R, Q, wires, estimation_wires, target_wires):
        """Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \\dots O_n.



        .. seealso:: :meth:`~.QuantumMonteCarlo.decomposition`.

        Args:
            A (array): unitary matrix corresponding to an input probability distribution
            R (array): unitary that encodes the function applied to the ancilla qubit register
            Q (array): matrix that encodes the expectation value according to the probability unitary
                and the function-encoding unitary
            wires (Any or Iterable[Any]): full set of wires that the operator acts on
            target_wires (Iterable[Any]): the target wires
            estimation_wires (Iterable[Any]): the estimation wires

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = [QubitUnitary(A, wires=target_wires[:-1]), QubitUnitary(R, wires=target_wires), qml.templates.QuantumPhaseEstimation(Q, target_wires=target_wires, estimation_wires=estimation_wires)]
        return op_list