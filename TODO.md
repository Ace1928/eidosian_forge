# Eidosian Documentation Forge TODO

## 0. Critical Pre-requisites
- [x] Investigate and get `eidos_mcp` running to enable tool usage. (Running on port 8928)

## 1. Project Setup & Scaffolding
- [x] Create a new `docs_forge` directory to house all the logic and generated documentation.
- [x] Create a `staging` directory inside `docs_forge` for human/AI review.
- [x] Create a `final_docs` directory inside `docs_forge` for the approved, published documentation.
- [x] Create a `scripts` directory inside `docs_forge` for the processing scripts.

## 2. Local LLM Setup (Constraint: < 3B Parameters)
- [x] Research and select a suitable small (< 3B params), agentic, tool-calling LLM via web search. (Chosen Model: Qwen 2.5 1.5B Instruct GGUF)
- [x] Research and document the steps to build `llama.cpp` from source. (Build from source using cmake: https://github.com/ggerganov/llama.cpp)
- [x] Write a script to download and build `llama.cpp`.
- [x] Write a script to download the selected < 3B LLM model. (Downloaded: qwen2.5-1.5b-instruct-q5_k_m.gguf)

## 3. Documentation Generation Loop
- [x] Create a script to scan the `eidosian_forge` repository directory-by-directory. (Index saved to file_index.json)
- [x] **CRITICAL:** Explicitly exclude `archive_forge` (too huge) and `.git`.
- [ ] For each identified file, create a prompt for the local LLM to generate:
    - File Summary
    - API Documentation
    - Current Status
    - Potential Future Directions
- [ ] Implement the logic to call the local `llama.cpp` model with the generated prompt and save the output to the `staging` directory.

## 4. Review & Approval Workflow
- [x] Create a script that lists all files in the `staging` directory. (review_docs.py)
- [ ] For each file, present it to the user (or a reviewer agent) for approval.
- [ ] If approved, move the file to the `final_docs` directory.
- [ ] If rejected, the user/agent provides feedback for regeneration.

## 5. HTML Index Generation
- [x] Create a script that generates a single `index.html` file in the root of the `eidosian_forge`. (generate_html_index.py)
- [ ] This HTML file should contain a navigable tree structure linking to the docs.

## 6. Automated Gating & Quality Checks
- [x] Implement robust, adaptive pre-checks (Markdown linting, placeholder detection). (Part of process_files.py)
- [ ] Implement a retry loop: if checks fail, feed errors back to the LLM for correction.
