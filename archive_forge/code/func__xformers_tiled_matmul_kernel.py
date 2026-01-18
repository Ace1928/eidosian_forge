import itertools
from typing import List, Tuple
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
@triton.autotune(configs=TRITON_CONFIGS, key=['M1', 'M2', 'M3', 'N1', 'N2', 'N3', 'K1', 'K2', 'K3'], prune_configs_by={'early_config_prune': our_early_config_prune, 'perf_model': our_estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'EVEN_K': lambda args: all((k % (args['BLOCK_K'] * args['SPLIT_K']) == 0 for k in [args['K1'], args['K2'], args['K3']]))})
@triton.jit()
def _xformers_tiled_matmul_kernel(A11, A12, A13, A21, A22, A23, A31, A32, A33, B11, B12, B13, B21, B22, B23, B31, B32, B33, C11, C12, C13, C21, C22, C23, C31, C32, C33, M1, M2, M3, N1, N2, N3, K1, K2, K3, stride_am1, stride_am2, stride_am3, stride_ak1, stride_ak2, stride_ak3, stride_bk1, stride_bk2, stride_bk3, stride_bn1, stride_bn2, stride_bn3, stride_cm1, stride_cm2, stride_cm3, stride_cn1, stride_cn2, stride_cn3, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, ACC_TYPE: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m1 = tl.cdiv(M1, BLOCK_M)
    grid_m2 = tl.cdiv(M2, BLOCK_M)
    grid_m3 = tl.cdiv(M3, BLOCK_M)
    grid_n1 = tl.cdiv(N1, BLOCK_N)
    grid_n2 = tl.cdiv(N2, BLOCK_N)
    grid_n3 = tl.cdiv(N3, BLOCK_N)
    grid_m = grid_m1 + grid_m2 + grid_m3
    grid_n = grid_n1 + grid_n2 + grid_n3
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    A1 = tl.where(pid_m < grid_m1, A11, tl.where(pid_m < grid_m1 + grid_m2, A21, A31))
    A2 = tl.where(pid_m < grid_m1, A12, tl.where(pid_m < grid_m1 + grid_m2, A22, A32))
    A3 = tl.where(pid_m < grid_m1, A13, tl.where(pid_m < grid_m1 + grid_m2, A23, A33))
    B1 = tl.where(pid_n < grid_n1, B11, tl.where(pid_n < grid_n1 + grid_n2, B12, B13))
    B2 = tl.where(pid_n < grid_n1, B21, tl.where(pid_n < grid_n1 + grid_n2, B22, B23))
    B3 = tl.where(pid_n < grid_n1, B31, tl.where(pid_n < grid_n1 + grid_n2, B32, B33))
    C = tl.where(pid_m < grid_m1, tl.where(pid_n < grid_n1, C11, tl.where(pid_n < grid_n1 + grid_n2, C12, C13)), tl.where(pid_m < grid_m1 + grid_m2, tl.where(pid_n < grid_n1, C21, tl.where(pid_n < grid_n1 + grid_n2, C22, C23)), tl.where(pid_n < grid_n1, C31, tl.where(pid_n < grid_n1 + grid_n2, C32, C33))))
    M = tl.where(pid_m < grid_m1, M1, tl.where(pid_m < grid_m1 + grid_m2, M2, M3))
    N = tl.where(pid_n < grid_n1, N1, tl.where(pid_n < grid_n1 + grid_n2, N2, N3))
    stride_ak = tl.where(pid_m < grid_m1, stride_ak1, tl.where(pid_m < grid_m1 + grid_m2, stride_ak2, stride_ak3))
    stride_bk = tl.where(pid_n < grid_n1, stride_bk1, tl.where(pid_n < grid_n1 + grid_n2, stride_bk2, stride_bk3))
    stride_cn = tl.where(pid_m < grid_m1, stride_cn1, tl.where(pid_m < grid_m1 + grid_m2, stride_cn2, stride_cn3))
    stride_cm = tl.where(pid_n < grid_n1, stride_cm1, tl.where(pid_n < grid_n1 + grid_n2, stride_cm2, stride_cm3))
    pid_m = tl.where(pid_m < grid_m1, pid_m, tl.where(pid_m < grid_m1 + grid_m2, pid_m - grid_m1, pid_m - grid_m1 - grid_m2))
    pid_n = tl.where(pid_n < grid_n1, pid_n, tl.where(pid_n < grid_n1 + grid_n2, pid_n - grid_n1, pid_n - grid_n1 - grid_n2))
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    grid_k1 = tl.cdiv(K1, BLOCK_K)
    grid_k2 = tl.cdiv(K2, BLOCK_K)
    grid_k3 = tl.cdiv(K3, BLOCK_K)
    for tile in range(pid_k, grid_k1 + grid_k2 + grid_k3, SPLIT_K):
        A = tl.where(tile < grid_k1, A1, tl.where(tile < grid_k1 + grid_k2, A2, A3))
        B = tl.where(tile < grid_k1, B1, tl.where(tile < grid_k1 + grid_k2, B2, B3))
        K = tl.where(tile < grid_k1, K1, tl.where(tile < grid_k1 + grid_k2, K2, K3))
        stride_am = tl.where(tile < grid_k1, stride_am1, tl.where(tile < grid_k1 + grid_k2, stride_am2, stride_am3))
        stride_bn = tl.where(tile < grid_k1, stride_bn1, tl.where(tile < grid_k1 + grid_k2, stride_bn2, stride_bn3))
        my_tile = tl.where(tile < grid_k1, tile, tl.where(tile < grid_k1 + grid_k2, tile - grid_k1, tile - grid_k1 - grid_k2))
        rk = my_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        Ain = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        Bin = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        if EVEN_K:
            a = tl.load(Ain)
            b = tl.load(Bin)
        else:
            a = tl.load(Ain, mask=rk[None, :] < K, other=0.0)
            b = tl.load(Bin, mask=rk[:, None] < K, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
    acc = acc.to(C.dtype.element_ty)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)