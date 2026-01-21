# Eidosian Forge - Dynamic Evolution TODO

## Priority Matrix
| Task | Filepath | Impact | Complexity | Severity | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Audit & Cleanse archive_forge | eidosian_forge/archive_forge | 8 | 4 | High | IN_PROGRESS (EXTERNAL) |
| Audit & Cleanse article_forge | eidosian_forge/article_forge | 7 | 4 | High | DONE |
| Standardize repo_forge | eidosian_forge/repo_forge | 9 | 6 | High | DONE |
| Review crawl_forge | eidosian_forge/crawl_forge | 6 | 5 | Med | DONE |
| Fix diagnostics_router timer | eidosian_forge/eidos_mcp/routers/diagnostics.py | 5 | 2 | Low | DONE |
| Refactor forge_loader | eidosian_forge/eidos_mcp/forge_loader.py | 10 | 3 | High | DONE |

## Derived Tasks (Derivation Log)
- **2026-01-21**: Identified massive file count in root (810k+). Focus shifted to entropy management in legacy forges.
- **2026-01-21**: Identity Protocol updated to Continuous Execution. Terminating "conclude session" logic.